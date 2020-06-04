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
import kornia
import torchvision
from torch import nn
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from eval import box_loss, calculate_image_precision

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from apex import amp
except ImportError:
    amp = None

from network import Network
from wheat_dataset import dataFactory, dataset_img

seq = nn.Sequential(kornia.augmentation.RandomHorizontalFlip(),
                    kornia.augmentation.RandomVerticalFlip(),
                    #kornia.augmentation.Normalize(0.5, 0.5)
                    )


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class EarlyStopping:
    def __init__(self, patience=5, cmp='small', e=0.01, metric='loss'):
        self.steps = patience
        self.best = None
        self.cmp = cmp  # small is best or large is best
        self.count = 0
        self.e = e
        self.metric = metric

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


class Save_best_weight:
    def __init__(self, path='best_weight.pth', cmp='small', metric='loss'):
        self.best = None
        self.cmp = cmp  # small is best or large is best
        self.metric = metric
        self.bestWeight = None
        self.path = path

    def __call__(self, val, engine):
        if self.best is None:
            self.best = val
            self.bestWeight = engine.model.state_dict()

        if self.cmp == 'small':
            if val < self.best:
                self.best = val
                self.bestWeight = engine.model.state_dict()
        else:
            if val > self.best:
                self.best = val
                self.bestWeight = engine.model.state_dict()

        if self.bestWeight:
            torch.save(self.bestWeight, self.path)
        return self.bestWeight


class CheckPoint:
    def __init__(self, path='checkPoint.pth', level=1):
        # level 0: end of batch
        # level 1: end of train loop
        # level 2: end of epoch
        # level 3: end of train
        self.path = path
        self.level = level

    def __call__(self, engine):
        checkpoint = {
            'model': engine.model.state_dict(),
            'optimizer': engine.optimizer.state_dict(),
        }
        if engine.FP16:
            checkpoint['amp'] = amp.state_dict()
        torch.save(checkpoint, self.path)


class LightHouse:
    def __init__(self, model: Module, optimizer, criterion,
                 lr_scheduler=None,
                 patience=None,
                 batchSize=5,
                 shuffle=False,
                 metric=None,
                 callback=None,
                 FP16=False,
                 opt_level='O1',
                 verbose=1):
        self.model = model
        self.optimizer = optimizer
        self.best_weight = None
        self.currentEpoch_weight = None
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.patience = patience
        self.verbose = verbose
        self.batchSize = batchSize
        self.shuffle = shuffle
        self.FP16 = FP16
        self.opt_level = opt_level
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.trainLoader = None
        self.valLoader = None
        self.testLoader = None
        self.trainProcessBar = None
        self.valProcessBar = None
        self.callback = callback

        self.metric = metric

        self.currentLoss = 0
        self.current_avg_loss = 0
        self.current_iou = 0
        self.total_iou = []
        self.total_loss = 0.0
        self.count = 0
        self.processBarInfo = dict()

        self.trainEndCallbacks = []
        self.valEndCallbacks = []
        self.epochEndCallbacks = []

    @classmethod
    def fastCreator(cls, param: dict):
        # dict={
        # 'model':'efficientNet_3B',
        # 'pretrained':true,
        # 'opt': 'SGD',
        # 'loss':'cross_entropy,
        # 'lr':0.0001,
        # 'momentum':0.9,
        # 'classes: 1,
        # }
        try:
            # get model
            modelName = param['model']
            if isinstance(modelName, str):
                pretrained = param.get('pretrained', False)
                model = cls.getModel(modelName, pretrained)
            elif isinstance(modelName, Module):
                model = modelName
            else:
                raise fastCreatorError('get no model')

            # get optimizer
            opt = param.get('optimizer', None)
            opt = opt if opt is not None else param.get('opt', None)
            if isinstance(opt, str):
                optimizer = cls.__getOptimizer(opt, model, param)
            else:
                raise fastCreatorError('get no model')

            # get criterion
            criterion = cls.__getCriterion(param['loss'])

            param.pop('model', None)
            param.pop('optimizer', None)
            param.pop('opt', None)
            param.pop('loss', None)
            return LightHouse(model, optimizer, criterion, **param)

        except KeyError as e:
            print(e)
        except fastCreatorError as e:
            print('get param error: ', e)

    @classmethod
    def getModel(cls, modelName: str, pretrained=False):
        model = None
        if modelName.startswith('efficientNet'):
            model = cls.__getEfficientNet(modelName, pretrained)
        if model is None:
            raise fastCreatorError('get model failed')
        return model

    @classmethod
    def __getEfficientNet(cls, modelName: str, pretrained):  # 'efficientnet_b0'
        try:
            model = torchLoad('rwightman/gen-efficientnet-pytorch', modelName, pretrained=pretrained)
            return model
        except RuntimeError:
            raise fastCreatorError('modelName:%s error' % modelName)

    @classmethod
    def __getOptimizer(cls, opt, model, param):
        if opt in ['SGD', 'sgd']:
            optimizer = SGD
        elif opt in ['Adam', 'adam']:
            optimizer = Adam
        else:
            raise fastCreatorError('optimizer name error')

        kwargs = dict()
        for k in ['lr', 'momentum']:
            val = param.get(k, None)
            if val:
                kwargs[k] = val
        return optimizer(params=model.parameters(), **kwargs)

    @classmethod
    def __getCriterion(cls, lossName):
        # L1Loss
        # MSELoss
        # CrossEntropyLoss
        # CTCLoss
        # NLLLoss
        # PoissonNLLLoss
        # KLDivLoss
        # BCELoss
        # BCEWithLogitsLoss
        # MarginRankingLoss
        # HingeEmbeddingLoss
        # MultiLabelMarginLoss
        # SmoothL1Loss
        # SoftMarginLoss
        # MultiLabelSoftMarginLoss
        # CosineEmbeddingLoss
        # MultiMarginLoss
        # TripletMarginLoss
        return getattr(torch.nn, lossName)()

    @staticmethod
    def getLoss(engine):
        return engine.current_avg_loss

    def train(self, trainLoader, valLoader=None, epoch=10):
        self.model.to(self.device)

        if self.FP16:
            self.prepareFP16()
        # process metric
        self.processMetric()
        # process callback
        self.processCallBacks()

        for i in range(1, epoch + 1):
            for phase in ['train','val']: #
                if phase == 'train':
                    self.train_loop(i, trainLoader)
                    self.end_train_loop(i, trainLoader)
                    self.runCallBacks(self.trainEndCallbacks)
                    # print train error
                elif phase == 'val':
                    self.val_loop(i, valLoader)
                    self.end_val_loop(i, valLoader)
                    self.runCallBacks(self.valEndCallbacks)
            # end epoch callback
            # save current weight
            weight = self.model.state_dict()
            torch.save(weight, './result/training_weight.pth')

        # before exit
        if self.best_weight:
            self.model.load_state_dict(self.best_weight)
        return self.model

    def prepareFP16(self):
        self.model, self.optimizer = amp.initialize(self.model,
                                                    self.optimizer,
                                                    opt_level=self.opt_level,
                                                    verbosity=0
                                                    )
        self.lr_scheduler = self.lr_scheduler.update(self.optimizer)

    def load_weight(self, path):
        self.model.load_state_dict(torch.load(path))

    def load_checkPoint(self, path, opt_level='O1'):
        checkpoint = torch.load(path)
        if self.FP16:
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level=opt_level)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
        else:
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def end_train_loop(self, epoch, trainLoader):
        l = len(trainLoader.dataset)
        epoch_loss = self.total_loss / l
        self.trainProcessBar.write('train loss: %.3f\n' %epoch_loss)

    def end_val_loop(self, epoch, valLoader):
        l = len(valLoader.dataset)
        epoch_loss = self.total_loss / l
        avg_iou = self.current_iou
        self.valProcessBar.write('val loss: %.3f\n'
                                 'avg iou: %.3f\n' %(epoch_loss, avg_iou))

    def exit_train(self):
        if self.best_weight:
            self.model.load_state_dict(self.best_weight)
        return self.model

    def exit_val(self):
        if self.best_weight:
            self.model.load_state_dict(self.best_weight)
        return self.model

    def runCallBacks(self, callbacks, phase='train'):
        if phase == 'train':
            endProcess = self.exit_train
        else:
            endProcess = self.exit_val
        for callback in callbacks:
            if isinstance(callback, EarlyStopping):
                if self.earlyStopping(callback):
                    return endProcess()
            elif isinstance(callback, Save_best_weight):
                self.best_weight = self.save_weight(callback)
            elif isinstance(callback, CheckPoint):
                callback(self)

    def processCallBacks(self):
        if self.callback:
            for call in self.callback:
                if isinstance(call, EarlyStopping):
                    if call.metric.startswith('val_'):
                        self.valEndCallbacks.append(call)
                    else:
                        self.trainEndCallbacks.append(call)
                elif isinstance(call, Save_best_weight):
                    if call.metric.startswith('val_'):
                        self.valEndCallbacks.append(call)
                    else:
                        self.trainEndCallbacks.append(call)
                elif isinstance(call, CheckPoint):
                    if call.level == 1:
                        self.trainEndCallbacks.append(call)
                    elif call.level == 2:
                        self.epochEndCallbacks.append(call)

    def earlyStopping(self, earlyStop: EarlyStopping):
        if earlyStop.metric.startswith('val_'):
            key = earlyStop.metric.split('_')[1]
        else:
            key = earlyStop.metric
        metricFunc = self.metric[key]
        val = metricFunc(self)
        return earlyStop(val)

    def save_weight(self, saveWeight: Save_best_weight):
        if saveWeight.metric.startswith('val_'):
            key = saveWeight.metric.split('_')[1]
        else:
            key = saveWeight.metric
        metricFunc = self.metric[key]
        val = metricFunc(self)
        return saveWeight(val, self)

    def getDataLoader(self, trainDataSet, valDataSet=None):
        train_params = {
            'batch_size': self.batchSize,
            'shuffle': self.shuffle,
            'num_workers': 4,
        }

        val_params = {
            'batch_size': self.batchSize,
            'shuffle': False,
            'num_workers': 4,
        }

        trainLoader = DataLoaderX(trainDataSet, **train_params)
        if valDataSet:
            valLoader = DataLoaderX(valDataSet, **val_params)
        else:
            valLoader = None
        return trainLoader, valLoader

    def predict(self, testLoader):
        pass

    def train_loop(self, epoch, dataloader):
        self.model.train()
        self.trainProcessBar = tqdm(total=len(dataloader), file=sys.stdout, leave=True)
        self.trainProcessBar.set_description('Train %d' % epoch)
        self.trainProcessBar.write('Epoch %d ......' % epoch)
        # statistics for category
        self.total_loss = 0.0
        self.count = 0

        for X, y in dataloader:
            X = list(img.to(self.device) for img in X)
            y = [{k: v.to(self.device) for k, v in t.items()} for t in y]

            with torch.set_grad_enabled(True):
                loss_dict = self.model(X, y)
                loss = sum(l for l in loss_dict.values())
                # backward + optimize
                
                self.optimizer.zero_grad()
                if self.FP16:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            # statistics
            self.total_loss += loss.item() * len(X)

            self.count += len(X)
            self.current_avg_loss = loss.item()

            # callback for train loop
            self.updateProcessBar(self.trainProcessBar)

    def val_loop(self, epoch, dataloader):
        self.model.eval()
        self.valProcessBar = tqdm(total=len(dataloader), file=sys.stdout, leave=True)
        self.valProcessBar.set_description('Val %d' % epoch)

        # statistics for category
        self.total_loss = 0.0
        self.count = 0
        self.total_iou = []

        for X, y in dataloader:
            X = list(img.to(self.device) for img in X)
            y = [{k: v.to(self.device) for k, v in t.items()} for t in y]

            with torch.set_grad_enabled(False):
                output = self.model(X, y)
                preds = list(map(lambda x: x['boxes'].cpu().numpy(), output))
                scores = list(map(lambda x: x['scores'].cpu().numpy(), output))
                gt_boxes = list(map(lambda x: x['boxes'].cpu().numpy(), y))

                for p, g, s in zip(preds, gt_boxes, scores):
                    preds_sorted_idx = np.argsort(s)[::-1]
                    p = p[preds_sorted_idx]
                    p[:, 0],p[:, 1],p[:, 2],p[:, 3] = p[:, 0],p[:,1],p[:,2]-p[:,0],p[:,3]-p[:,1]
                    g[:, 0],g[:, 1],g[:, 2],g[:, 3] = g[:, 0],g[:, 1],g[:, 2]-g[:, 0],g[:, 3]-g[:, 1]
                    iou = calculate_image_precision(p, g)
                    self.total_iou.append(iou)

                # loss = sum(l for l in output.values())
            # statistics
            self.total_loss = np.sum(self.total_iou)

            self.count += len(X)
            self.current_avg_loss = np.mean(self.total_iou)
            self.current_iou = np.mean(self.total_iou)

            # callback for val loop
            self.updateProcessBar(self.valProcessBar)

    def updateProcessBar(self, processBar):
        if processBar is self.trainProcessBar:
            self.processBarInfo['loss'] = self.current_avg_loss
        else:
            for key, func in self.metric.items():
                self.processBarInfo[key] = func(self)

        processBar.set_postfix(**self.processBarInfo)
        processBar.update()

    def processMetric(self):
        if self.metric is None:
            self.metric = {
                'loss': self.getLoss
            }

    def summary(self):
        pass

    def callback(self, event='LOOP_END'):  # callback function
        # TRAIN_LOOP_END
        # VAL_LOOP_END
        # LOOP_END
        # PREDICT_END
        # CALLBACK  == EPOCH_END
        # EPOCH_END
        def decorator(func, *args, **kwargs):
            self.add2CallList(event, func, args, kwargs)

        return decorator

    def add2CallList(self, event, func, *args, **kwargs):
        if event == 'PROCESS_BAR':
            self.epochEndCallbacks.append([event, func, args, kwargs])



class fastCreatorError(RuntimeError):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg


###########################################################

def weighted_auc(y_true, y_valid):
    tpr_thresholds = [0, 0.4, 1.0]
    weights = [2, 1]
    # print(y_true, y_valid)

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    if tpr[0] == np.nan or fpr[0] == np.nan:
        return 0
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min <= tpr) & (tpr <= y_max)
        # print(y_min, y_max, tpr, mask)
        if len(fpr[mask])==0:
            return 0
        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization

class FP16_lr_scheduler:
    def __init__(self, lr_scheduler, **kwargs):
        self.lr_scheduler = lr_scheduler
        self.kwargs = kwargs

    def update(self, optimizer):
        self.kwargs['optimizer'] = optimizer
        return self.lr_scheduler(**self.kwargs)


def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == '__main__':
    d = dataFactory()
    # set params
    BATCH_SIZE = 8
    learning_rate = 1e-3
    patience = 3
    e = 0.001
    sample = 20000

    def customLoss(engine: LightHouse):
        return engine.current_avg_loss


    def customIOU(engine: LightHouse):
        return np.mean(engine.total_iou)

    channels = 3
    classes = 2

    #train_X, train_y, val_X, val_y = d.getBinary(sample, frac=0.85)
    train_X, train_y, val_X, val_y = d.getALL(frac=0.9)
    print(train_X.shape, train_y.shape, val_X.shape, val_y.shape)
    best_weight_path = './result/best_weight.pth'

    # trainSet = dataset_img(train_X, train_y, binary=True)
    # valSet = dataset_img(val_X, val_y, binary=True)
    trainSet = dataset_img(train_X, train_y)
    valSet = dataset_img(val_X, val_y, isTrain=False)
    #trainSet = dataset_stack2(train_X, train_y, binary=True)
    #valSet = dataset_stack2(val_X, val_y, binary=True)

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

    model = Network(num_classes=classes, pretrained=True)

    criterion = CrossEntropyLoss()
    # criterion = BCEWithLogitsLoss()
    # optimizer = SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = ExponentialLR(optimizer, gamma=0.975)
    # lr_scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5,patience=2, verbose=False, eps=1e-6)
    # lr_scheduler = FP16_lr_scheduler(ReduceLROnPlateau,
    #                                  optimizer=optimizer,
    #                                  mode='min',
    #                                  factor=0.5,
    #                                  patience=2,
    #                                  verbose=False,
    #                                  eps=1e-6)
    # lr_scheduler = FP16_lr_scheduler(ExponentialLR, optimizer=optimizer, gamma=0.975)
    metric = {
        'loss': customLoss,
        'iou': customIOU,
    }

    # callbacks
    earlyStop = EarlyStopping(patience=patience, metric='val_loss', cmp='large')
    saveWeight = Save_best_weight(path=best_weight_path, metric='val_loss', cmp='large')
    checkPoint = CheckPoint(path = './result/checkpoint.pth')
    # logger = Logger(file='xxxxxx/logFile.log') # log everything to a file, txt or pickle dict


    engine = LightHouse(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        batchSize=BATCH_SIZE,
        metric=metric,
        FP16=False,
        opt_level='O1',
        callback=[earlyStop, saveWeight, checkPoint],
    )
    # engine.load_weight(saveWeight.path)
    if os.path.exists('./result/training_weight.pth'):
        try:
            engine.load_weight('./result/training_weight.pth')
        except Exception as e:
            print('load training weight failed')

    epoch = 30
    print('BATCH_SIZE:', BATCH_SIZE)
    print('epoch:', epoch)
    model = engine.train(trainLoader, valLoader, epoch=epoch)
