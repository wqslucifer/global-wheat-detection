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
import cv2
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import torchvision
from torch import nn
from eval import box_loss, calculate_image_precision, find_best_match
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from network import Network
from torchvision import transforms
from wheat_dataset import dataFactory, dataset_img, dataset_img_test
from matplotlib.pyplot import plot as plt
import warnings
from ensemble_boxes import *
from trainModel import LightModel

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def collate_fn(batch):
    return tuple(zip(*batch))


def run_wbf(predictions, image_index, image_size=512, iou_thr=0.55, skip_box_thr=0.7, weights=None):
    boxes = [prediction[image_index]['boxes'].data.cpu().numpy() / (image_size - 1) for prediction in predictions]
    scores = [prediction[image_index]['scores'].data.cpu().numpy() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]) for prediction in predictions]
    boxes, scores, labels = weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr,
                                                  skip_box_thr=skip_box_thr)
    boxes = boxes * (image_size - 1)
    return boxes, scores, labels


if __name__ == '__main__':
    print('load data')
    d = dataFactory()
    test_X = d.getTest()
    # trainSet = dataset_img(train_X[:10], train_y[:10])

    testSet = dataset_img_test(test_X)

    testLoader = DataLoaderX(testSet,
                             batch_size=8,
                             shuffle=False,
                             num_workers=os.cpu_count(),
                             collate_fn=collate_fn
                             )

    model = LightModel(num_classes=2)
    model.load_state_dict(torch.load('./result/weight.pth'))
    device = torch.device('cuda:0')
    model.to(device)
    model.eval()

    results = []

    with torch.set_grad_enabled(False):
        for j, (x, x_id) in enumerate(testLoader):
            if j > 0:
                break
            x = list(img.to(device) for img in x)
            predictions = []
            _, outputs = model(x, None)
            predictions.append(outputs)

            boxes = list(map(lambda x: x['boxes'].cpu().numpy(), outputs))
            scores = list(map(lambda x: x['scores'].cpu().numpy(), outputs))
            labels = list(map(lambda x: x['labels'].cpu().numpy(), outputs))

            image_size = 1024
            iou_thr = 0.55
            skip_box_thr = 0.5  # exclude boxes with score lower than this variable

            for i in range(len(outputs)):
                b, s, l = weighted_boxes_fusion(
                    [boxes[i]], [scores[i]], [labels[i]],
                    weights=None,
                    iou_thr=iou_thr,
                    skip_box_thr=skip_box_thr)

                image_id = x_id[i]

                str_list = []
                b[:, 2] = b[:, 2] - b[:, 0]
                b[:, 3] = b[:, 3] - b[:, 1]
                for box, score in zip(b, s):
                    str_list.append('%.4f %f %f %f %f' % (score, box[0], box[1], box[2], box[3]))
                box_str = " ".join(str_list)

                result = {'image_id': image_id, 'PredictionString': box_str}
                results.append(result)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.head()

def plot(images, ):
    sample = images[1].permute(1,2,0).cpu().numpy()
    boxes = outputs[1]['boxes'].data.cpu().numpy()
    scores = outputs[1]['scores'].data.cpu().numpy()

    boxes = boxes[scores >= 0.5].astype(np.int32)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    for box in boxes:
        cv2.rectangle(sample,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (220, 0, 0), 2)

    ax.set_axis_off()
    ax.imshow(sample)