import os
import gc
import cv2
import datetime
import torch
import json
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmenters import meta
from imgaug import parameters as iap
from imgaug.augmentables import bbs
from sklearn.model_selection import StratifiedKFold

from albumentations import RandomBrightnessContrast, HueSaturationValue, Compose, OneOf

GREEN = [0, 255, 0]
ORANGE = [255, 140, 0]
RED = [255, 0, 0]

np.random.seed(4869)


def draw_bbs(image, bbs):
    for bb in bbs.bounding_boxes:
        if bb.is_fully_within_image(image.shape):
            color = RED
        elif bb.is_partly_within_image(image.shape):
            color = ORANGE
        else:
            color = GREEN
        image = bb.draw_on_image(image, size=2, color=color, alpha=0.85)
    return image


def sometimes(aug): return iaa.Sometimes(0.7, aug)
def sometimes_p9(aug): return iaa.Sometimes(0.9, aug)
def sometimes_p5(aug): return iaa.Sometimes(0.5, aug)


simple_seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        sometimes_p5(iaa.Rot90(k=ia.ALL)),
        # sometimes_p5(iaa.Cutout(nb_iterations=2, size=0.05, cval=0)),
    ],
    random_order=True
)

seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        sometimes_p5(iaa.Cutout(nb_iterations=5, size=0.05, cval=0)),
        sometimes_p5([
            iaa.CropToFixedSize(width=1024-50, height=1024-50),
            # iaa.CropToFixedSize(width=900, height=900),
            iaa.Resize((1024, 1024)),
        ]),
        sometimes(iaa.Rot90(k=ia.ALL)),
    ],
    random_order=True
)

album_seq = Compose([
    OneOf([
        HueSaturationValue(p=1),
        RandomBrightnessContrast(p=1),
    ], p=1)
], p=0.5)


class randomMerge(meta.Augmenter):
    def __init__(self, p=1.0,
                 seed=None, name=None, trainPath=None, labels=None,
                 random_state="deprecated", deterministic="deprecated"):
        super(randomMerge, self).__init__(
            seed=seed, name=name)
        self.p = iap.handle_probability_param(p, "p")
        self.seed = seed
        self.random_state = None
        self.deterministic = None
        self.trainPath = trainPath
        self.labels = labels
        self.randomCenterX = 0
        self.randomCenterY = 0
        self.cropTopLeft = None
        self.cropTopRight = None
        self.cropBottomLeft = None
        self.cropBottomRight = None
        self.imageSize = (1024, 1024)
        self.boxes = []
        self.seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 20% of all images
                sometimes_p5(iaa.Rot90(k=ia.ALL)),
            ],
            random_order=False
        )

    def _augment_batch_(self, batch, random_state, parents, hooks):
        samples = self.p.draw_samples((batch.nb_rows,),
                                      random_state=random_state)
        for i, sample in enumerate(samples):
            if sample >= 0.5:
                if batch.images is not None:
                    thisImg = batch.images[i]
                    target = np.zeros(thisImg.shape).astype('uint8')
                    thisBoxes = batch.bounding_boxes[i]
                    targetBoxes = []
                    randomImgs, randomBoxes = self.getImages(
                        thisImg, thisBoxes)
                    # for idx in range(4):
                    #     randomImgs[idx] = album_seq(
                    #         image=randomImgs[idx])['image']
                    #     randomImgs[idx], randomBoxes[idx] = self.seq(
                    #         image=randomImgs[idx], bounding_boxes=randomBoxes[idx])

                    topLeftImg, topLeftBoxes = self.cropTopLeft(image=randomImgs[0],
                                                                bounding_boxes=randomBoxes[0])
                    topRightImg, topRighBoxes = self.cropTopRight(image=randomImgs[1],
                                                                  bounding_boxes=randomBoxes[1])
                    bottomLeftImg, bottomLeftBoxes = self.cropBottomLeft(image=randomImgs[2],
                                                                         bounding_boxes=randomBoxes[2])
                    bottomRightImg, bottomRightBoxes = self.cropBottomRight(image=randomImgs[3],
                                                                            bounding_boxes=randomBoxes[3])
                    target[:self.randomCenterY,
                           :self.randomCenterX, :] = topLeftImg
                    target[:self.randomCenterY,
                           self.randomCenterX:, :] = topRightImg
                    target[self.randomCenterY:,
                           :self.randomCenterX, :] = bottomLeftImg
                    target[self.randomCenterY:,
                           self.randomCenterX:, :] = bottomRightImg
                    targetBoxes += topLeftBoxes.remove_out_of_image().clip_out_of_image().bounding_boxes
                    targetBoxes += topRighBoxes.remove_out_of_image().clip_out_of_image().shift(
                        x=self.randomCenterX).bounding_boxes
                    targetBoxes += bottomLeftBoxes.remove_out_of_image().clip_out_of_image().shift(
                        y=self.randomCenterY).bounding_boxes
                    targetBoxes += bottomRightBoxes.remove_out_of_image().clip_out_of_image().shift(
                        x=self.randomCenterX, y=self.randomCenterY).bounding_boxes

                    bboxList = []
                    threshold = 30
                    for b in targetBoxes:
                        if b.x2 - b.x1 < threshold or b.y2 - b.y1 < threshold:
                            continue
                        else:
                            bboxList.append(b)

                    batch.images[i] = target
                    batch.bounding_boxes[i] = bbs.BoundingBoxesOnImage(
                        bboxList, shape=target.shape)
        return batch

    def getImages(self, thisImg, thisBoxes):
        self.randomCenterX = np.random.randint(100, 924)
        self.randomCenterY = np.random.randint(100, 924)
        self.cropTopLeft = iaa.Crop(
            px=(0, 1024 - self.randomCenterX, 1024 - self.randomCenterY, 0),
            keep_size=False,
        )  # top, right, bottom, left
        self.cropTopRight = iaa.Crop(
            px=(0, 0, 1024 - self.randomCenterY, self.randomCenterX),
            keep_size=False
        )  # top, right, bottom, left
        self.cropBottomLeft = iaa.Crop(
            px=(self.randomCenterY, 1024 - self.randomCenterX, 0, 0),
            keep_size=False
        )  # top, right, bottom, left
        self.cropBottomRight = iaa.Crop(
            px=(self.randomCenterY, 0, 0, self.randomCenterX),
            keep_size=False
        )  # top, right, bottom, left
        randomImgs = [thisImg]
        randomBoxes = [thisBoxes]
        for _ in range(3):
            index = np.random.choice(range(len(self.trainPath)))
            path = self.trainPath[index]
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            boxes = self.labels[index]

            b_list = []
            for box in boxes:
                b = bbs.BoundingBox(
                    x1=box[0], x2=box[0] + box[2], y1=box[1], y2=box[1] + box[3])
                b_list.append(b)
            bbox_img = bbs.BoundingBoxesOnImage(b_list, shape=img.shape)

            randomImgs.append(img)
            randomBoxes.append(bbox_img)
        return randomImgs, randomBoxes

    def get_parameters(self):
        """See :func:`~imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.p, self.seed]


class dataset_img(Dataset):
    def __init__(self, trainPath, labels, isTrain=True):
        self.trainPath = trainPath
        self.labels = labels
        self.isTrain = isTrain
        # self.resize = iaa.Resize((512, 512))
        self.resize = iaa.Resize((1024, 1024))
        # self.cropResize = iaa.CropToFixedSize(width=512, height=512)
        self.randomZip = randomMerge(p=0.8, trainPath=trainPath, labels=labels)
        self.randomZip_1 = randomMerge(p=1, trainPath=trainPath, labels=labels)

    def __len__(self):
        return len(self.trainPath)

    def __getitem__(self, index):
        path = self.trainPath[index]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox = self.labels[index]
        b_list = []
        for box in bbox:
            b = bbs.BoundingBox(
                x1=box[0], x2=box[0] + box[2], y1=box[1], y2=box[1] + box[3])
            b_list.append(b)
        bbox_img = bbs.BoundingBoxesOnImage(b_list, shape=img.shape)

        count = 0
        while True:
            if self.isTrain:
                if count > 20:
                    img_aug, bbox_aug = simple_seq(
                        image=img, bounding_boxes=bbox_img)
                else:
                    # if np.random.choice([True, False], p=[0.5, 0.5]):
                    #     img, bbox_img = self.randomZip_1(
                    #         image=img, bounding_boxes=bbox_img)

                    img = album_seq(
                        image=img)['image']
                    img_aug, bbox_aug = seq(
                        image=img, bounding_boxes=bbox_img)
                        
                img_aug, bbox_aug = self.resize(
                    image=img_aug, bounding_boxes=bbox_aug)

                # filter out boxes less than 20
                bboxList = []
                threshold = 20
                for b in bbox_aug.remove_out_of_image().clip_out_of_image().bounding_boxes:
                    if b.x2 - b.x1 < threshold or b.y2 - b.y1 < threshold:
                        continue
                    else:
                        bboxList.append(b)
                bbox_aug = bbs.BoundingBoxesOnImage(
                    bboxList, shape=img_aug.shape)
                b_list = bbox_aug.remove_out_of_image().clip_out_of_image().bounding_boxes
            else:
                img_aug, bbox_aug = self.resize(
                    image=img, bounding_boxes=bbox_img)
                b_list = bbox_aug.bounding_boxes

            if len(b_list)>0:
                break
            else:
                count += 1

        bbox = np.array(
            list(map(lambda x: [x.x1, x.y1, x.x2, x.y2], b_list)), dtype=np.float32)
        img_aug = np.transpose(img_aug, [2, 0, 1]).astype(np.float32)
        img_aug /= 255.0

        label = dict()
        label['boxes'] = torch.stack(
            tuple(map(torch.tensor, zip(*bbox)))).permute(1, 0)
        label['labels'] = torch.ones((len(bbox),), dtype=torch.int64)
        img_aug = torch.tensor(img_aug, dtype=torch.float)

        return img_aug, label


class dataset_img_gray(Dataset):
    def __init__(self, trainPath, labels, isTrain=True):
        self.trainPath = trainPath
        self.labels = labels
        self.isTrain = isTrain

    def __len__(self):
        return len(self.trainPath)

    def __getitem__(self, index):
        path = self.trainPath[index]
        img = cv2.imread(path)
        b, g, r = cv2.split(img)
        gray = (0.441 * r + 0.811 * g + 0.385 * b + 18.7874).astype(np.uint8)
        img = cv2.merge((gray, gray, gray))

        bbox = self.labels[index]
        b_list = []
        for box in bbox:
            b = bbs.BoundingBox(
                x1=box[0], x2=box[0] + box[2], y1=box[1], y2=box[1] + box[3])
            b_list.append(b)
        bbox_img = bbs.BoundingBoxesOnImage(b_list, shape=img.shape)

        while True:
            if self.isTrain:
                img_aug, bbox_aug = seq(image=img, bounding_boxes=bbox_img)
                break
            else:
                img_aug, bbox_aug = img, bbox_img
                b_list = bbox_aug.remove_out_of_image().clip_out_of_image().bounding_boxes
                if len(b_list):
                    break

        bbox = np.array(
            list(map(lambda x: [x.x1, x.y1, x.x2, x.y2], b_list)), dtype=np.float32)
        img_aug = img_aug.astype(np.float32)
        img_aug = np.transpose(img_aug, [2, 0, 1])
        img_aug /= 255.0
        # area = (bbox[:, 3] - bbox[:, 1]) * (bbox[:, 2] - bbox[:, 0])
        # area = torch.as_tensor(area, dtype=torch.float32)

        label = dict()
        # label['boxes'] = bbox
        label['boxes'] = torch.stack(
            tuple(map(torch.tensor, zip(*bbox)))).permute(1, 0)
        label['labels'] = torch.ones((len(bbox),), dtype=torch.int64)
        # label['iscrowd'] = torch.zeros((len(bbox),), dtype=torch.int64)
        # label['area'] = area

        img_aug = torch.tensor(img_aug, dtype=torch.float)

        return img_aug, label


class dataset_img_test(Dataset):
    def __init__(self, trainPath):
        self.trainPath = trainPath

    def __len__(self):
        return len(self.trainPath)

    def __getitem__(self, index):
        path = self.trainPath[index]
        img = cv2.imread(path)
        img = cv2.resize(img, (1024, 1024))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = np.transpose(img, [2, 0, 1])
        img /= 255.0
        img = torch.tensor(img, dtype=torch.float)
        return img, os.path.basename(path)[:-4]


class dataFactory:
    def __init__(self):
        self.dataPath = '/home/lucifer/project/data/global-wheat-detection/'
        self.projectPath = '/home/lucifer/project/global-wheat-detection/'
        self.train_csv = pd.read_csv(os.path.join(self.dataPath, 'train.csv'))
        tmp1 = self.train_csv.groupby('image_id')['source'].apply(
            lambda x: list(x)[0]).reset_index()
        tmp2 = self.train_csv.groupby('image_id')['bbox'].apply(
            lambda x: list(json.loads(i) for i in x)).reset_index()
        self.img_box_csv = pd.merge(tmp1, tmp2, on='image_id')
        self.submission_csv = pd.read_csv(
            os.path.join(self.dataPath, 'sample_submission.csv'))
        self.kFold = StratifiedKFold(
            n_splits=10, shuffle=True, random_state=100)

    def getALL(self, frac=0.8):
        l = len(self.img_box_csv)
        trainLen = int(l * frac)
        # valLen = l-trainLen

        train_val_list = []
        for trainIndex, valIndex in self.kFold.split(self.img_box_csv, self.img_box_csv.source):
            train_val_list.append((trainIndex, valIndex))

        # trainIndex = np.random.choice(range(l), trainLen, replace=False)
        # valIndex = np.setdiff1d(range(l), trainIndex)

        trainIndex, valIndex = train_val_list[0]

        trainPaths = np.array(
            self.img_box_csv['image_id'].apply(lambda x: os.path.join(self.dataPath, 'train', x + '.jpg')))
        boxList = np.array(self.img_box_csv['bbox'])

        # trainIndex = trainIndex[:20]

        return trainPaths[trainIndex], \
            boxList[trainIndex], \
            trainPaths[valIndex], \
            boxList[valIndex]

    def getTest(self):
        testPath = self.submission_csv.image_id.apply(
            lambda x: os.path.join(self.dataPath, 'test', x + '.jpg'))
        return testPath
