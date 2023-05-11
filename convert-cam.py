#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2023/4/26 10:16
# !@Author  : murInj
# !@Filer    : .py
import os
import torch
from proccess import modelHeat
import cv2
from mobileNet import MobileNetV3_Small

"""
This file converts all the images inside the img folder into CAM heatmaps into the heatMap folder
"""

"""PARAMS"""
en_glcm = True
glcm_inv = True
img_dir = 'img'
cam_dir = 'heatMap'
model_path = 'model/mobile800k.pth'

model = torch.load(model_path)
model.eval()


SIZE = None

if __name__ == "__main__":
    for name in os.listdir(cam_dir):
        os.remove(os.path.join(cam_dir, name))
        print("remove {}".format(name))

    fileNames = os.listdir(img_dir)
    print("proccess {} files".format(len(fileNames)))
    for name in fileNames:
        # try:
            img = cv2.imread(os.path.join(img_dir, name), 1)[:, :, ::-1]
            H, W, C = img.shape
            visualization = modelHeat(img,model,SIZE,en_glcm,glcm_inv)
            cv2.imwrite(os.path.join(cam_dir, name), visualization)
            print("write {}".format(name))
        # except:
        #     print("process {} failed".format(name))
