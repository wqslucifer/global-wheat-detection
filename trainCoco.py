import os
import sys
import gc
import torch
import time
import numpy as np
import pandas as pd
from prefetch_generator import BackgroundGenerator
from torch.hub import load as torchLoad
from torch.nn import Module, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss
from torch.nn import Conv2d, Module, BatchNorm2d, ReLU, Dropout, AdaptiveAvgPool2d, Linear, Sequential
from torch.nn import functional as F
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torchvision
from torch import nn
from eval import box_loss, calculate_image_precision, find_best_match
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, \
    fasterrcnn_resnet50_fpn, FasterRCNN, resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from torchvision import transforms
from wheat_dataset import dataFactory, dataset_img, dataset_img_gray
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, ProgressBar
from pytorch_lightning import loggers
from ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion
from imgaug import augmenters as iaa
from imgaug.augmentables import bbs


import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def collate_fn(batch):
    return tuple(zip(*batch))


def get_fasterrcnn_resnet50_fpn(num_classes=2, pretrained=True, pretrained_backbone=True,
                                trainable_backbone_layers=5):
    model = fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        pretrained_backbone=pretrained_backbone,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


class LightModel(LightningModule):
    def __init__(self, channels=3, num_classes=2, lr=1e-3, pretrained=True):
        super().__init__()
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # resnet101 = torchvision.models.resnet101(pretrained=True)
        # resnetLayers = list(resnet101.children())[:-1]
        # backbone_resnet101 = nn.Sequential(*resnetLayers)
        # backbone_resnet101.out_channels = 2048

        backbone_resnet_fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            'resnet101',
            pretrained=pretrained,
            trainable_layers=5,
        )
        #########################################

        self.model = FasterRCNN(backbone_resnet_fpn,
                                num_classes=num_classes)
        self.t = transforms.ToTensor()

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = list(self.t(img).to(self.device) for img in x)
        y = y # [{k: v for k, v in t[0].items()} for t in y]
        loss_dict = self(x, y)
        loss = sum(l for l in loss_dict.values())
        # loss /= len(batch)
        return {'loss': loss}

    def training_epoch_end(self, outputs):
        avg_loss = np.stack([x['loss'].item() for x in outputs]).mean()
        # print(avg_iou, avg_loss)
        # progress_bar = {'loss': avg_loss}
        tensorboard_logs = {'train_loss': avg_loss}
        self.logger.log_metrics(tensorboard_logs, self.current_epoch)
        torch.save(self.model.state_dict(), './result/weight_coco_epoch_%d.pth' % self.current_epoch)
        return {'loss': avg_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = list(self.t(img).to(self.device) for img in x)
        y = y

        loss_dict, result = self(x, y)
        loss = sum(l.cpu().numpy() for l in loss_dict.values())
        # loss /= len(batch)

        preds = list(map(lambda a: a['boxes'].cpu().numpy(), result))
        scores = list(map(lambda a: a['scores'].cpu().numpy(), result))
        labels = list(map(lambda a: a['labels'].cpu().numpy(), result))
        gt_boxes = list(map(lambda a: a['boxes'].cpu().numpy(), y))

        ious = 0
        for p, g, s, l in zip(preds, gt_boxes, scores, labels):
            b, s, l = weighted_boxes_fusion(
                        [p], [s], [l],
                        weights=None,
                        iou_thr=0.55,
                        skip_box_thr=0.5)

            preds_sorted_idx = np.argsort(s)[::-1]
            b = b[preds_sorted_idx]
            index = np.where(s >= 0.5)
            p = b[index]
            # p[:, 0], p[:, 1], p[:, 2], p[:, 3] = p[:, 0], p[:, 1], p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
            # g[:, 0], g[:, 1], g[:, 2], g[:, 3] = g[:, 0], g[:, 1], g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]
            ious += calculate_image_precision(g, p, thresholds=(0.55,), form='coco')
        return {'val_loss': loss, 'val_IOU': ious}

    def validation_epoch_end(self, outputs):
        avg_iou = np.stack([x['val_IOU'] for x in outputs]).mean()
        avg_loss = np.stack([x['val_loss'] for x in outputs]).mean()
        # print(avg_iou, avg_loss)
        tensorboard_logs = {'val_loss': avg_loss, 'val_IOU': avg_iou}
        self.logger.log_metrics(tensorboard_logs, self.current_epoch)
        return {'val_loss': avg_loss, 'val_IOU': avg_iou}

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.01)
        # scheduler = ExponentialLR(opt, gamma=0.975)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=1, threshold=3e-4, min_lr=1e-6)
        return [opt], [scheduler]


if __name__ == '__main__':
    BATCH_SIZE = 2

    coco_train = torchvision.datasets.CocoDetection(
        root='../data/coco/train2017',
        annFile='../data/coco/annotations/instances_train2017.json',
    )

    coco_val = torchvision.datasets.CocoDetection(
        root='../data/coco/val2017',
        annFile='../data/coco/annotations/instances_val2017.json',
    )

    trainLoader = DataLoaderX(coco_train,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=os.cpu_count(),
                              collate_fn=collate_fn
                              )
    valLoader = DataLoaderX(coco_val,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=os.cpu_count(),
                            collate_fn=collate_fn,
                            )

    tb_logger = loggers.TensorBoardLogger('logs/')

    model = LightModel(num_classes=80,
                       lr=1e-3,
                       pretrained=True)
    trainer = Trainer(gpus=1,
                      max_epochs=25,
                      # accumulate_grad_batches={1: 2, 8: 4},
                      accumulate_grad_batches={1: 2},
                      # auto_scale_batch_size='binsearch',
                      # use_amp=True
                      # amp_level='O1',
                      # auto_lr_find=False,
                      # early_stop_callback=early_stop_callback,
                      # checkpoint_callback=checkpoint_callback,
                      logger=tb_logger,
                      weights_summary='top',
                      )

    # model.load_state_dict(torch.load('./result/weight_1024_epoch_29.pth'))
    trainer.fit(model,
                train_dataloader=trainLoader,
                val_dataloaders=valLoader,
                )