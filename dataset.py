#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2023/4/26 10:42
# !@Author  : murInj
# !@Filer    : .py
import json
import os
from proccess import Process,clear
import cv2
from torch.utils.data import Dataset
import numpy as np

class camDataset(Dataset):
    def __init__(self, img_dir,inputSize,load=False):
        self.inputSize = inputSize
        self._img_dir = img_dir
        self.fileNames = os.listdir(self._img_dir)
        self.data = []
        self.labels = []
        n = img_dir.split('/')[-1]
        if load is False:
            print("load data from dir")
            for name in self.fileNames:
                try:
                    img = cv2.resize(cv2.imread(os.path.join(self._img_dir, name), 1)[:, :, ::-1],(inputSize[1],inputSize[2]))
                    self.data.append(np.array(img, dtype=np.float32))
                    self.labels.append(Process(img,(inputSize[1],inputSize[2]),glcm_on=False,heat=False))
                    print("load {}".format(name))
                except:
                    continue
            self.labels = np.expand_dims(self.labels, axis=0)
            np.save("data#"+n+'.npy',self.data)
            np.save("labels#" + n + '.npy',self.labels)
        else:
            print("load data from np")
            self.data = np.load("data#"+n+'.npy')
            self.labels = np.load("labels#" + n + '.npy')
        print("data loaded")
        clear()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        return X, y
