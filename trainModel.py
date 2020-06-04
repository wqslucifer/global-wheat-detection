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
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torchvision
from torch import nn
from eval import box_loss, calculate_image_precision, calculate_precision
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
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # resnet101 = torchvision.models.resnet101(pretrained=True)
        # resnetLayers = list(resnet101.children())[:-1]
        # backbone_resnet101 = nn.Sequential(*resnetLayers)
        # backbone_resnet101.out_channels = 2048

        # backbone_resnet_fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        #     'resnet101',
        #     pretrained=True,
        #     trainable_layers=5,
        # )
        #########################################
        self.model = get_fasterrcnn_resnet50_fpn(num_classes=2,
                                                 pretrained=True,
                                                 pretrained_backbone=True,
                                                 trainable_backbone_layers=5)

        # self.model = FasterRCNN(backbone_resnet_fpn,
        #                         num_classes=2)

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = list(img.to(self.device) for img in x)
        # y = [{k: v.to(self.device) for k, v in t.items()} for t in y]
        x = list(img for img in x)
        y = [{k: v for k, v in t.items()} for t in y]
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
        torch.save(self.model.state_dict(),
                   './result/weight_1024_epoch_%d.pth' % self.current_epoch)
        # torch.save(self.model.state_dict(),
        #            './result/weight_512_epoch_%d.pth' % self.current_epoch)
        return {'loss': avg_loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = list(img for img in x)
        y = [{k: v for k, v in t.items()} for t in y]
        _, result = self(x, None)
        #loss = sum(l.cpu().numpy() for l in loss_dict.values())
        # loss /= len(batch)

        preds = list(map(lambda a: a['boxes'].cpu().numpy(), result))
        scores = list(map(lambda a: a['scores'].cpu().numpy(), result))
        labels = list(map(lambda a: a['labels'].cpu().numpy(), result))
        gt_boxes = list(map(lambda a: a['boxes'].cpu().numpy(), y))

        iouList = []
        for p, g, s, l in zip(preds, gt_boxes, scores, labels):
            # box, score, _ = weighted_boxes_fusion(
            #     [p], [s], [l],
            #     weights=None,
            #     iou_thr=0.5,
            #     skip_box_thr=0.5)
            index = np.where(s >= 0.5)
            p = p[index]
            s = s[s >= 0.5]
            box, score, _ = p, s, l
            
            preds_sorted_idx = np.argsort(score)[::-1]
            box = box[preds_sorted_idx]

            # p[:, 0], p[:, 1], p[:, 2], p[:, 3] = p[:, 0], p[:, 1], p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]
            # g[:, 0], g[:, 1], g[:, 2], g[:, 3] = g[:, 0], g[:, 1], g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]
            ious = np.ones((len(g), len(box))) * -1
            # print(calculate_precision(g.copy(), box, threshold=0.5, form='not coco', ious=ious))
            iouList.append(calculate_precision(
                g.copy(), box, threshold=0.5, form='not coco', ious=ious))
        return {'val_IOU': np.mean(iouList)}

    def validation_epoch_end(self, outputs):
        avg_iou = np.stack([x['val_IOU'] for x in outputs]).mean()
        # avg_loss = np.stack([x['val_loss'] for x in outputs]).mean()
        # print(avg_iou, avg_loss)
        tensorboard_logs = {'val_IOU': avg_iou}
        self.logger.log_metrics(tensorboard_logs, self.current_epoch)
        return {'val_loss': avg_iou, 'val_IOU': avg_iou}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        # opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.01)
        # scheduler = ExponentialLR(opt, gamma=0.975)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.3, patience=2, threshold=3e-4, min_lr=1e-7)
        # scheduler = MultiStepLR(opt, milestones=[10, 20, 30], gamma=0.3)
        return [opt], [scheduler]


if __name__ == '__main__':
    d = dataFactory()

    BATCH_SIZE = 6
    learning_rate = 3e-4
    patience = 3
    e = 0.001
    sample = 20000
    channels = 3
    classes = 2

    train_X, train_y, val_X, val_y = d.getALL(frac=1)

    print(train_X.shape, train_y.shape)
    print(val_X.shape, val_y.shape)

    trainSet = dataset_img(train_X, train_y)
    # trainSet = dataset_img(train_X, train_y, isTrain=False)
    valSet = dataset_img(val_X, val_y, isTrain=False)

    # trainSet = dataset_img_gray(train_X, train_y)
    # valSet = dataset_img_gray(val_X, val_y, isTrain=False)

    trainLoader = DataLoaderX(trainSet,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=os.cpu_count(),
                              collate_fn=collate_fn
                              )
    valLoader = DataLoaderX(valSet,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=os.cpu_count(),
                            collate_fn=collate_fn
                            )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=6,
        verbose=True,
        mode='min'
    )
    checkpoint_callback = ModelCheckpoint(
        filepath='./result/{epoch}',
        # filepath='./result/checkpoint.pth',
        monitor='val_loss',
        save_weights_only=True,
        mode='min',
        # verbose=True,
    )
    tb_logger = loggers.TensorBoardLogger('logs/')

    model = LightModel(num_classes=classes,
                       lr=learning_rate,
                       pretrained=True)
    trainer = Trainer(gpus=1,
                      max_epochs=50,
                      # accumulate_grad_batches={10:2, 15:3, 20:4, 25:5, 30:6},
                      # accumulate_grad_batches={45: 2},
                      # auto_scale_batch_size='binsearch',
                      # use_amp=True
                      # amp_level='O1',
                      # auto_lr_find=False,
                      # early_stop_callback=early_stop_callback,
                      # checkpoint_callback=checkpoint_callback,
                      logger=tb_logger,
                      weights_summary='top',
                      )

    # model.model.load_state_dict(torch.load('./result/weight_1024_epoch_27.pth'))
    trainer.fit(model,
                train_dataloader=trainLoader,
                val_dataloaders=valLoader,
                )

    torch.save(model.state_dict(), './result/weight.pth')

    '''
    No.1:
        kFold-5_1 
        augmentation: None
        fasterRCNN resnet101_fpn backbone pretrained trainable_layers:5 
        batchSize:4 accumulate:4 
        epoch: 9 
        predict: 1024
        local loss: 0.4526
        val IOU: 0.797
        LB: 0.5902
    No.2:
        kFold-5_1 
        augmentation: None
        fasterrcnn_resnet50_fpn pretrained trainable_layers:5 
        batchSize:8 accumulate:2 
        epoch: 7
        predict: 1024
        local loss: 0.6143
        val IOU: 1.567
        LB: 0.6163
    No.3:
        kFold-5_1 
        augmentation: None
        fasterrcnn_resnet50_fpn pretrained trainable_layers:3
        batchSize:8 accumulate:2
        predict: 1024
        local loss: same as trainable_layers:5
    No.4:
        kFold-5_1 
        augmentation: 0.5*(HFlip, VFlip), 0.9*(AddToHueAndSaturation, AddToBrightness)
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:2
        epoch: 19
        predict: 1024
        local loss: 0.6281
        val IOU: 1.1165
        LB: 0.6457
    No.5:
        kFold-5_1 
        input size: 512
        augmentation: 0.5*(HFlip, VFlip), 0.9*(AddToHueAndSaturation, AddToBrightness)
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:2
        epoch: 19
        predict: 512
        local loss: 0.6132
        val IOU: 0.7513
        LB: 0.6313
    No.6:
        kFold-5_1 
        input size: 1024
        augmentation: full augmentation
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:2
        epoch: 18
        predict: 1024
        local loss: 0.6394
        val IOU: 1.158
        LB: 0.6409
        
        ensamble: 1024+512, iou_thr: 0.55, score_thr:0.5, LB: 0.6441
    No.7:
        kFold-5_1 
        input size: 1024
        augmentation: 0.5*(HFlip, VFlip), 0.9*(AddToHueAndSaturation, AddToBrightness),
        0.9*(Hue, Bright), 0.5* Cutout
        
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:2
        epoch: 19
        predict: 1024
        local loss: 
        val IOU: 
        LB:0.6386
    No.8:
        kFold-5_1 
        input size: 1024
        augmentation: full aug
        fasterrcnn_resnet101_fpn pretrained trainable_layers: 5
        batchSize:4 accumulate:4
        epoch: 19
        predict: 1024
        local loss: 0.66
        val IOU: 
        LB:  0.5X
    No.9:
        kFold-5_1 
        input size: 1024
        augmentation: full augmentation + custom merge
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:2
        epoch: 
        predict: 1024
        local loss: 0.633
        val IOU: 1.18
        LB: 0.6402
        
        ensamble: 
    No.10:
        kFold-5_1 
        input size: 1024
        augmentation: full augmentation + custom merge
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:2
        epoch: 19
        predict: 1024
        local loss: 0.644
        val IOU: 1.175
        LB: 0.6256
        
        ensamble: 
    No.11:
        kFold-5_1 
        input size: 1024
        augmentation: full augmentation + custom merge new aug
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:2
        epoch: 
        predict: 1024
        local loss: 0.6154
        val IOU: 1.171
        LB: 0.6446
        
        ensamble: 0.6467
    No.12:
        kFold-5_1 
        input size: 1024
        augmentation: full augmentation + more merge and cutout
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:2
        epoch: 
        predict: 1024
        local loss: 0.295
        val IOU: 0.815
        LB: 0.6742
        ensamble: 
    No.13:
        kFold-10_1 
        input size: 1024
        augmentation: full augmentation
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:1
        epoch: 35
        predict: 1024
        local loss: 0.3608
        val IOU: 0.8209
        LB: 0.6800
        ensamble: 
    No.14:
        kFold-10_1 
        input size: 1024
        augmentation: full augmentation
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6 accumulate:1
        epoch:44
        predict: 1024
        local loss:0.3364
        val IOU:0.8237
        LB: 0.6899
        ensamble: 
    No.15:
        kFold-10_1 
        input size: 1024
        augmentation: full augmentation
        fasterrcnn_resnet101_fpn pretrained trainable_layers: 5
        batchSize:4 accumulate:1
        epoch: 45
        predict: 1024
        local loss:0.346
        val IOU:0.817
        LB: 0.67
        ensamble: 
    No.16: from No.14
        kFold-10_1 
        input size: 1024
        augmentation: full augmentation --> + add more mixup img, more crop, add CLAHE to all img
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6  accumulate: + 45: 2
        epoch: 35
        predict: 1024
        local loss:0.2746
        val IOU:0.8022
        LB: 0.63
        ensamble: 
    No.17: from No.16
        kFold-10_1 
        input size: 1024
        augmentation: full augmentation --> - CLAHE to all img + seq in merge aug
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6  accumulate: - 45: 2
        epoch: 
        predict: 1024
        local loss:0.2782
        val IOU:0.815
        LB: 0.6796
        ensamble: 
    No.18: from No.17
        kFold-10_1 
        input size: 512
        augmentation: full augmentation --> + add album aug, new hue and brightcontrast
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6  accumulate: 1
        epoch: 
        predict: 512
        local loss:
        val IOU:
        LB: 0.67
        ensamble:
    No.19: from No.18
        kFold-10_1 
        input size: 1024
        augmentation: full augmentation
        fasterrcnn_resnet50_fpn pretrained trainable_layers: 5
        batchSize:6  accumulate: 1
        epoch: 
        predict: 1024
        local loss:
        val IOU:
        LB: 0.67
        ensamble:
    '''
