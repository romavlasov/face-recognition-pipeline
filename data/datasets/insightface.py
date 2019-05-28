import os
import pickle
import datetime

import numpy as np
import pandas as pd

import torch
import cv2

import torch.utils.data as data


class Train(data.Dataset):
    def __init__(self, folder, dataset, transforms=None):
        self.folder = os.path.join(folder, dataset)
        self.annotation = pd.read_csv(os.path.join(folder,
                                                   dataset,
                                                   'train.csv'))
        self.transforms = transforms

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        np.random.seed(datetime.datetime.now().microsecond)
        item = self.annotation.iloc[idx]

        image = self._get_image(item['image'])

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(item['label'])
    
    def _get_image(self, image_name):
        image = cv2.imread(os.path.join(self.folder, 'train', image_name))
        return image / 255.


class Test(data.Dataset):
    def __init__(self, folder, dataset, transforms=None):
        self.path = os.path.join(folder, dataset + '.bin')
        self.data, self.labels = pickle.load(open(self.path, 'rb'),
                                             encoding='bytes')
        self.transforms = transforms
        
    def __getitem__(self, index):
        image1 = self._decode(self.data[index * 2])
        image2 = self._decode(self.data[index * 2 + 1])
        label = self.labels[index]
        
        if self.transforms:
            image1 = self.transforms(image1)
            image2 = self.transforms(image2)
        
        return image1, image2, label
    
    def _decode(self, data):
        image = np.asarray(bytearray(data), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image / 255.
        
    def __len__(self):
        return len(self.labels)
