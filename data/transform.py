import cv2
import torch

import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class OneOf(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, image):
        transform = np.random.choice(self.transforms)
        image = transform(image)
        return image
    
    
class RandomApply(object):
    def __init__(self, transforms, prob=0.5):
        self.transforms = transforms
        self.prob = prob    
        
    def __call__(self, image):
        for t in self.transforms:
            if np.random.rand() < self.prob:
                image = t(image)
        return image
    

class ToTensor(object):
    def __call__(self, image):
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return image.float()
    

class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = np.array(mean or [0.485, 0.456, 0.406])
        self.std = np.array(std or [0.229, 0.224, 0.225])

    def __call__(self, image):
        image = (image - self.mean) / self.std
        return image


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        image = cv2.resize(image, (self.size, self.size))
        return image


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        height, width = image.shape[:2]
        if isinstance(self.output_size, int):
            new_height, new_width = self.output_size, self.output_size
        else:
            new_height, new_width = self.output_size

        s_height = int(height / 2 - new_height / 2)
        s_width = int(width / 2 - new_width / 2)
        image = image[s_height:s_height + new_height, 
                      s_width:s_width + new_width]
        return image


class RandomContrast(object):
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, image):
        alpha = np.random.uniform(1.0 - self.delta, 1.0 + self.delta)
        image *= alpha
        image = np.clip(image, 0, 1)
        return image


class RandomBrightness(object):
    def __init__(self, delta=0.1):
        self.delta = delta

    def __call__(self, image):
        delta = np.random.uniform(-self.delta, self.delta)
        image += delta
        image = np.clip(image, 0, 1)
        return image


class GaussianBlur(object):
    def __init__(self, kernel=3):
        self.kernel = (kernel, kernel)
    
    def __call__(self, image):
        image = cv2.blur(image, self.kernel)
        return image


class RandomRotation(object):
    def __init__(self, angle=10, aligne=False):
        self.angle = angle
        self.aligne = aligne
        
    def __call__(self, image):        
        angle = np.random.uniform(-self.angle, self.angle)

        height, width = image.shape[:2]
        cX, cY = width / 2, height / 2
     
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)

        if self.aligne:
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
     
            width = int((height * sin) + (width * cos))
            height = int((height * cos) + (width * sin))
     
            M[0, 2] += (width / 2) - cX
            M[1, 2] += (height / 2) - cY
     
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT_101)
        return image


class RandomShift(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio
        
    def __call__(self, image):
        height, width = image.shape[:2]
        
        horizontal = np.random.randint(-height * self.ratio, height * self.ratio)
        vertival = np.random.randint(-width * self.ratio, width * self.ratio)
        
        M = np.float32([[1, 0, horizontal], 
                        [0, 1, vertival]])
        
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT_101)
        return image


class Expand(object):
    def __init__(self, ratio=0.9):
        self.ratio = ratio

    def __call__(self, image):
        height, width = image.shape[:2]
        ratio = np.random.uniform(self.ratio, 1.0)
    
        horizontal = np.random.uniform(height * (1.0 - ratio))
        vertical = np.random.uniform(width * (1.0 - ratio))
    
        M = np.float32([[ratio, 0, horizontal], 
                        [0, ratio, vertical]])
    
        image = cv2.warpAffine(image, M, (width, height), borderMode=cv2.BORDER_REFLECT_101)
        return image


class RandomShare(object):
    def __init__(self, ratio=0.1, aligne=True):
        self.ratio = ratio
        self.aligne = aligne
        
    def __call__(self, image):
        ratio = np.random.uniform(0.05, self.ratio)
        
        height, width = image.shape[:2]
        dx = int(ratio * width)
        
        box0 = np.array([[0,0], [width,0], [width,height], [0,height]], np.float32)
        
        dx1, dy1 = np.random.uniform(dx), np.random.uniform(dx)
        dx2, dy2 = np.random.uniform(dx), np.random.uniform(dx)
        dx3, dy3 = np.random.uniform(dx), np.random.uniform(dx)
        dx4, dy4 = np.random.uniform(dx), np.random.uniform(dx)
        
        box1 = np.array([[dx1, dy1], 
                         [width - dx2, dy2], 
                         [width - dx3, height - dy3], 
                         [dx4, height - dy4]], np.float32)
    
        mat = cv2.getPerspectiveTransform(box0, box1)
        
        image = cv2.warpPerspective(image, mat, (width, height),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REFLECT_101)
        
        if self.aligne:
            y1, y2 = box1[:, 1].min(), box1[:, 1].max()
            x1, x2 = box1[:, 0].min(), box1[:, 0].max()
            image = image[int(y1):int(y2), int(x1):int(x2)]
            
        image = cv2.resize(image, (width, height))
        
        return image


class HorizontalFlip(object):
    def __call__(self, image):
        image = cv2.flip(image, 1)
        return image


class VerticalFlip(object):  
    def __call__(self, image):
        image = cv2.flip(image, 0)
        return image


class SimultaneousFlip(object):  
    def __call__(self, image):
        image = cv2.flip(image, -1)
        return image


class Transforms(object):
    def __init__(self, input_size=112, train=True):
        self.train = train
        self.transforms = RandomApply([
            RandomApply([
                GaussianBlur(kernel=3),
            ], prob=0.2),
            HorizontalFlip(),
            RandomShift(ratio=0.1),
            RandomRotation(angle=10, aligne=False),
        ], prob=0.5)

        self.normalize = Compose([
            Resize(input_size),
            ToTensor(),
        ])

    def __call__(self, image):
        if self.train:
            image = self.transforms(image)
        image = self.normalize(image)
        return image
