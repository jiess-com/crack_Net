
from numpy import random

import numpy as np
import cv2

def t_random(min=0, max=1):
    return min + (max - min) * np.random.rand()


def t_randint(min, max):
    return np.random.randint(low=min, high=max)

class augCompose(object):

    def __init__(self, transforms=None):
        self.transforms = transforms

    def __call__(self, img, mask):

        if self.transforms is not None:
            for op, prob in self.transforms:
                if t_random() <= prob:
                    img, mask = op(img, mask)

        return img, mask

def Reshape(img,mask,size=448):
    return cv2.resize(img,(size,size)),cv2.resize(mask,(size,size))
#随机反转
def random_flip(img, label,p=0.5):
    if t_random() < p:
        i=np.random.randint(-1,1)
        img = np.flip(img,i)
        label = np.flip(label,i)

    return img, label

def RandomRomate(img,gt, p=0.5):
    if  t_random() < p:
        angle_list=[10,20,30,40,50,60,70,80,90]
        angle = angle_list[np.random.randint(0, 8)]
        # 定义旋转中心
        rows, cols = img.shape[:2]
        center = (cols / 2, rows / 2)
        # 定义旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1)
        # 进行旋转操作
        img = cv2.warpAffine(img, M, (cols, rows))
        gt = cv2.warpAffine(gt, M, (cols, rows))
    return img,gt

def RandomMove(img,gt,p=0.5):
    if t_random() < p:
        d = random.randint(0, 50)
        # print(d)
        # 定义平移矩阵
        M = np.float32([[1, 0, d], [0, 1, d]])

        # 进行平移操作
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        gt = cv2.warpAffine(gt, M, (gt.shape[1], gt.shape[0]))
    return img,gt

def RandomCropBoth(img, label, size=448):

    w, h = label.shape
    if w == size and h == size:
        return img, label

    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    image = img[y: y + size, x: x + size, :]
    # image = img.crop((x, y, x + size, y + size))
    label = label[y: y + size, x: x + size]

    # label = mask.crop((x, y, x + size, y + size))
    # edge = edge.crop((x, y, x + size, y + size))
    return image, label

def RandomBlur(img, mask):

    r = 5

    if t_random() < 0.2:
        return cv2.GaussianBlur(img,(r,r),0), mask

    if t_random() < 0.15:
        return cv2.blur(img,(r,r)), mask

    if t_random() < 0.1:
        return cv2.medianBlur(img,r), mask


    return img, mask

def RandomColorJitter(img, mask, brightness=32, contrast=0.5, saturation=0.5, hue=0.1,
                        prob=0.5):
    if brightness != 0 and t_random() > prob:
        img = _Brightness(img, delta=brightness)
    if contrast != 0 and t_random() > prob:
        img = _Contrast(img, var=contrast)
    if saturation != 0 and t_random() > prob:
        img = _Saturation(img, var=saturation)
    if hue != 0 and t_random() > prob:
        img = _Hue(img, var=hue)

    return img, mask


def _Brightness(img, delta=32):
    img = img.astype(np.float32) + t_random(-delta, delta)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _Contrast(img, var=0.3):
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    alpha = 1.0 + t_random(-var, var)
    img = alpha * img.astype(np.float32) + (1 - alpha) * gs
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _Hue(img, var=0.05):
    var = t_random(-var, var)
    to_HSV, from_HSV = [
        (cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2RGB),
        (cv2.COLOR_BGR2HSV, cv2.COLOR_HSV2BGR)][t_randint(0, 2)]
    hsv = cv2.cvtColor(img, to_HSV).astype(np.float32)

    hue = hsv[:, :, 0] / 179. + var
    hue = hue - np.floor(hue)
    hsv[:, :, 0] = hue * 179.

    img = cv2.cvtColor(hsv.astype('uint8'), from_HSV)
    return img


def _Saturation(img, var=0.3):
    gs = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gs = np.expand_dims(gs, axis=2)
    alpha = 1.0 + t_random(-var, var)
    img = alpha * img.astype(np.float32) + (1 - alpha) * gs.astype(np.float32)
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


