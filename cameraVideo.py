#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2023/4/16 14:13
# !@Author  : murInj
# !@Filer    : .py
import os
import time
import cv2
import numpy as np
import torch

from proccess import modelHeat


"""
This file will take the video image from the camera and convert it to a CAM heat map for display
"""

"""PARAMS"""
videoSRC = 0
# videoSRC = "v.mp4"
static_thresh = 5000
freeze_thresh = 10
ShowSize = (320, 240)
ProccessSize = (512, 512)
KeepCAM = False
cls = None
glcm_on = True
glcm_inv = False
font = cv2.FONT_HERSHEY_SIMPLEX
CAM_mode = False
information = True
model_path = 'model/mobile800k.pth'
resnet_path = 'model/resnet101.pth'

model = torch.load(model_path).eval()
resnet_model = torch.load(resnet_path).eval()
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


def patern(frame, origin, cam, glcm, glcm_inv, information):
    cv2.rectangle(frame, (0, 0), (512, 40), (0, 0, 0), -1)
    cv2.putText(frame, 'origin', (5, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, 'glcm', (125, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, 'cam', (235, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, 'glcm_inv', (335, 24), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if cam == True:
        cv2.circle(frame, (310, 18), 5, (255, 255, 255), -1)
    if origin == True:
        cv2.circle(frame, (100, 18), 5, (255, 255, 255), -1)
    if glcm == True:
        cv2.circle(frame, (210, 18), 5, (255, 255, 255), -1)
    if glcm_inv == True:
        cv2.circle(frame, (480, 18), 5, (255, 255, 255), -1)
    if information == True:
        # cv2.rectangle(frame, (0, 462), (512, 512), (0, 0, 0), -1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        entropy = round(imageEntropy(gray), 3)
        cv2.putText(frame, 'Entropy = '+f'{entropy}', (5, 471), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        topk = TOPK(frame, 3)
        cv2.putText(frame, 'TOPk = ' + f'{topk}', (5, 501), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if cam == True:
            Mr = round(MineralizationRatio(gray), 3)
            cv2.putText(frame, 'Mr = ' + f'{Mr}', (5, 441), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def TOPK(RGB_img, K):
    global resnet_model
    img = np.float32(np.expand_dims(RGB_img, axis=0))
    img = torch.tensor(img).permute(0, 3, 1, 2).cuda()
    outputs = resnet_model(img).cpu().detach().numpy()
    outputs = outputs[0]
    indices = np.argsort(outputs)
    top_k_indices = indices[-K:]
    top_k = outputs[top_k_indices]
    top_k = ((top_k + np.min(top_k)) * 0.5 / (np.max(top_k) + np.min(top_k)) + 0.5) * 0.96
    img = img.cpu()
    top_k = top_k[::-1]
    top_k_indices = top_k_indices[::-1]
    topK = list()
    for i in range(len(top_k)):
        topK.append(round(top_k[i], 3))
    return topK

def imageEntropy(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    entropy = 0
    for i in range(256):
        p = hist[i] / gray.size
        if p != 0:
            entropy += -p * np.log2(p)
    return float(entropy[0])

def MineralizationRatio(gray_heatMap, thres=0.5):
    shape = gray_heatMap.shape
    total = shape[0] * shape[1]
    ret, binary = cv2.threshold(gray_heatMap, thres * 255, 255, cv2.THRESH_BINARY)
    pixel_num = np.sum(binary == 0)
    return np.float32(pixel_num) / total

if __name__ == "__main__":
    cap = cv2.VideoCapture(videoSRC)
    if not cap.isOpened():
        print('Error opening camera')
    # 获取视频信息
    width = int(cap.get(3))
    height = int(cap.get(4))
    print(width,height)
    print(f'Save Path: {os.getcwd()}\\output.avi')
    # 创建VideoWriter对象,用于保存结果
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, ShowSize,True)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, ProccessSize)
            is_diff(frame)



            if ret == True:
                if CAM_mode or KeepCAM:
                    img = modelHeat(frame,model,ProccessSize,glcm_on,glcm_inv)
                    patern(img, False, CAM_mode or KeepCAM, glcm_on, glcm_inv, information)
                    img = cv2.resize(img, ShowSize)
                    out.write(img)
                    cv2.imshow('frame', img)
                else:
                    img = cv2.resize(frame, ProccessSize)
                    patern(img, True, False, False, False, information)
                    img = cv2.resize(img, ShowSize)
                    out.write(img)
                    cv2.imshow('frame', img)

            if cv2.waitKey(1) == ord('q'):
                break
            if cv2.waitKey(1) == ord('c'):
                CAM_mode = False if CAM_mode else True
            if cv2.waitKey(1) == ord('g'):
                glcm_on = False if glcm_on else True
            if cv2.waitKey(1) == ord('i'):
                glcm_inv = False if glcm_inv else True
            if cv2.waitKey(1) == ord('f'):
                information = False if information else True
            pre_frame = frame
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
