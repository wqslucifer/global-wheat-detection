from torch.nn import Conv2d, Module, BatchNorm2d, ReLU, Dropout, AdaptiveAvgPool2d, Linear, Sequential
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


import numpy as np
from efficientnet_pytorch import EfficientNet

class EarlyStopping:
    def __init__(self, epoch=5, cmp='small', e=0.01):
        self.steps = epoch
        self.best = None
        self.cmp = cmp  # small is best or large is best
        self.count = 0
        self.e = e

    def __call__(self, val):
        if self.best is None:
            self.best = val
            return False

        if self.cmp == 'small':
            if val < self.best and self.best - val > self.e:
                self.best = val
                self.count = 0
                return False
            else:
                self.count += 1
                if self.count == self.steps:
                    return True
                return False
        else:
            if val > self.best and val - self.best > self.e:
                self.best = val
                self.count = 0
                return False
            else:
                self.count += 1
                if self.count == self.steps:
                    return True
                return False


class Network(Module):
    def __init__(self, channels=3, num_classes = 2, pretrained=True):
        super(Network, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, pretrained_backbone=pretrained)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, x, y):
        return self.model(x, y)
