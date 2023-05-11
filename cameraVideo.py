#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2023/4/16 14:13
# !@Author  : murInj
# !@Filer    : .py
import time
import cv2
import numpy as np
import torch

from proccess import modelHeat,clear,processImg
clear()

"""
This file will take the video image from the camera and convert it to a CAM heat map for display
"""

"""PARAMS"""
static_thresh = 5000
freeze_thresh = 10
ShowSize = (640, 640)
ProccessSize = (512, 512)
KeepCAM = False
cls = None
glcm_on = True
glcm_inv = False
font = cv2.FONT_HERSHEY_SIMPLEX
CAM_mode = False
model_path = 'model/mobile800k.pth'

model = torch.load(model_path).eval()

pre_frame = None
static_cnt = 0


def is_diff(frame):
    global pre_frame
    global CAM_mode
    global static_cnt
    if pre_frame is None or frame is None:
        static_cnt = 0
        CAM_mode = False
        return

    diff = cv2.absdiff(frame, pre_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    thresh, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = [cv2.contourArea(c) for c in contours]

    # print(np.sum(area))
    if np.sum(area) < static_thresh:  # static frame
        static_cnt += 1
        if static_cnt > freeze_thresh:
            CAM_mode = True
    else:
        static_cnt = 0
        CAM_mode = False



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error opening camera')
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, ProccessSize)
        is_diff(frame)

        if ret == True:
            if CAM_mode or KeepCAM:
                img = modelHeat(frame,model,ProccessSize,glcm_on,glcm_inv)
                cv2.imshow('frame', img)
            else:

                img = cv2.resize(frame, ProccessSize)
                cv2.imshow('frame',img)
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('c'):
            CAM_mode = False if CAM_mode else True
        if cv2.waitKey(1) == ord('g'):
            glcm_on = False if glcm_on else True
        if cv2.waitKey(1) == ord('i'):
            glcm_inv = False if glcm_inv else True
        pre_frame = frame

    cap.release()
    cv2.destroyAllWindows()
