#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


########################### Data Augmentation ###########################



class Normalize(object):  # todo
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, nir):
        image = (image - self.mean) / self.std
        mask /= 255
        nir /= 255
        return image, mask, nir


class RandomCrop(object):
    def __call__(self, image, mask):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3]


class RandomCrop_edge(object):
    def __call__(self, image, mask, edge):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], edge[p0:p1, p2:p3]


class RandomFlip(object):
    def __call__(self, image, mask):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1]
        else:
            return image, mask


class RandomFlip_edge(object):
    def __call__(self, image, mask, edge):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1], edge[:, ::-1]
        else:
            return image, mask, edge


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, nir):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        nir = cv2.resize(nir, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, nir


class ToTensor(object):
    def __call__(self, image, mask, nir):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        nir = torch.from_numpy(nir)
        return image, mask, nir


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()

        self.randomcrop_edge = RandomCrop_edge()
        self.randomflip_edge = RandomFlip_edge()
        self.resize = Resize(352, 352)
        self.totensor = ToTensor()
        print('我的路径=>', cfg.datapath + '/' + cfg.mode + '.txt')
        with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name = self.samples[idx]
        # print('=>name', name)
        # image = cv2.imread(self.cfg.datapath + '/image/' + name + '.jpg')[:, :, ::-1].astype(np.float32)
        # print('=>show', name, ' ', self.cfg.datapath)
        # print('=show:', self.cfg.datapath + '/image/' + name + '.jpg')
        # print('=show:', self.cfg.datapath + '/mask/' + name + '.png')
        # print('=>path', self.cfg.datapath + '/image/' + name + '.jpg')
        # print('xxx:', '/image/' + name + '.jpg')

        image = cv2.imread(self.cfg.datapath + '/image/' + name + '.jpg')[:, :, ::-1].astype(np.float32)  # 原jpg
        # print('--', self.cfg.datapath.split('/')[-1])
        # if self.cfg.datapath.split('/')[-1] == 'VT821' or 'VT1000':
        if self.cfg.datapath.endswith('VT5000-Test') or self.cfg.datapath.endswith('VT5000-Train'):
            mask = cv2.imread(self.cfg.datapath + '/mask/' + name + '.png', 0).astype(np.float32)  # png
        else:
            # print('=>path', self.cfg.datapath + '/Mask/' + name + '.jpg')
            mask = cv2.imread(self.cfg.datapath + '/mask/' + name + '.jpg', 0).astype(np.float32)  # jpg
        # if self.cfg.datapath.split('/')[-1] == 'VT5000-Test':
        # mask = cv2.imread(self.cfg.datapath + '/Mask/' + name + '.png', 0).astype(np.float32)  # VT5000
        nir = cv2.imread(self.cfg.datapath + '/inf/' + name + '.jpg', 0).astype(np.float32)
        # if self.cfg.mode == 'train':
        #     edge = cv2.imread(self.cfg.datapath + '/edge/' + name + '_edge.png', 0).astype(np.float32)
        shape = mask.shape

        if self.cfg.mode == 'train':
            image, mask, nir = self.normalize(image, mask, nir)  # for s # todo
            # image, mask, edge = self.randomcrop_edge(image, mask, edge)
            # image, mask, edge = self.randomflip_edge(image, mask, edge)

            # image, edge = self.normalize(image, edge)  # for e
            # image, edge = self.randomcrop(image, edge)
            # image, edge = self.randomflip(image, edge)

            return image, mask, nir
        else:
            image, mask, nir = self.normalize(image, mask, nir)  # for s
            image, mask, nir = self.resize(image, mask, nir)
            image, mask, nir = self.totensor(image, mask, nir)

            # image, edge = self.normalize(image, edge)  # for e # todo dont need edge
            # image, edge = self.resize(image, edge)
            # image, edge = self.totensor(image, edge)
            return image, mask, nir, shape, name  # todo dont need edge

    def collate(self, batch):
        # size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        size = [352][np.random.randint(0, 1)]
        image, mask, nir = [list(item) for item in zip(*batch)]  # todo zzz

        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            nir[i] = cv2.resize(nir[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)  # todo zzz

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        nir = torch.from_numpy(np.stack(nir, axis=0)).unsqueeze(1)  # todo zzz

        return image, mask, nir  # todo zzz

    def __len__(self):
        return len(self.samples)


########################### Testing Script ###########################
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    plt.ion()

    cfg = Config(mode='train', datapath='../data/DUTS')
    data = Data(cfg)
    for i in range(1000):
        image, mask, edge = data[i]  # todo zzz
        image = image * cfg.std + cfg.mean
        plt.subplot(121)
        plt.imshow(np.uint8(image))
        plt.subplot(122)
        plt.imshow(mask)
        input()
