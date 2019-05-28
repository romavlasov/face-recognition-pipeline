import os
import pickle
import datetime

import numpy as np
import pandas as pd

import torch
import cv2

import torch.utils.data as data


class VGGFace2(data.Dataset):
    def __init__(self, folder, transforms=None):
        self.folder = folder
        self.annotation = self._get_annotation(folder)
        self.transforms = transforms

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        np.random.seed(datetime.datetime.now().microsecond)
        item = self.annotation.iloc[idx]

        image = self._get_image(item)
        width, height = image.shape[:2]
        
        boxes = self._get_boxes(item, width, height)
        landmarks = self._get_landmarks(item, width, height)
        target = self._get_target(item)

        if self.transforms:
            image = self.transforms(image)

        return image, np.concatenate((boxes, landmarks, target), axis=1)
    
    def _get_boxes(self, item, width, height):
        return np.array([[item['X'] / width,
                          item['Y'] / height,
                          (item['X'] + item['W']) / width,
                          (item['Y'] + item['H']) / height]], dtype=np.float)
    
    def _get_landmarks(self, item, width, height):
        return np.array([[item['P1X'] / width, item['P1Y'] / height,
                          item['P2X'] / width, item['P2Y'] / height,
                          item['P3X'] / width, item['P3Y'] / height,
                          item['P4X'] / width, item['P4Y'] / height,
                          item['P5X'] / width, item['P5Y'] / height]], dtype=np.float)
                                
    def _get_target(self, item):
        return [[1]]
    
    def _get_image(self, item):
        pass
    
    def _get_annotation(self, folder):
        pass


class Train(VGGFace2):
    def _get_image(self, item):
        image = cv2.imread(os.path.join(self.folder, 'train', item['NAME_ID'] + '.jpg'))
        return image / 255.

    def _get_annotation(self, folder):
        bb = pd.read_csv(os.path.join(folder, 'bb_landmark/loose_bb_train.csv'))
        landmark = pd.read_csv(os.path.join(folder, 'bb_landmark/loose_landmark_train.csv'))    
        return pd.merge(bb, landmark, on='NAME_ID')


class Test(VGGFace2):
    def _get_image(self, item):
        image = cv2.imread(os.path.join(self.folder, 'test', item['NAME_ID'] + '.jpg'))
        return image / 255.

    def _get_annotation(self, folder):
        bb = pd.read_csv(os.path.join(folder, 'bb_landmark/loose_bb_test.csv'))
        landmark = pd.read_csv(os.path.join(folder, 'bb_landmark/loose_landmark_test.csv'))    
        return pd.merge(bb, landmark, on='NAME_ID')