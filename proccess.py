#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2023/4/18 20:29
# !@Author  : murInj
# !@Filer    : .py
import cv2
import fast_glcm
import numpy as np
import torch
from pytorch_grad_cam import XGradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# model = torch.load('model/resnet101.pth').eval().cuda()
# targetLayer = [model.layer4[-1]]
#
# outputs = None
#
# def output_hook(model, input, output):
#     global outputs
#     outputs = output
#
# model.fc.register_forward_hook(output_hook)
#
# cam = XGradCAM(model=model, target_layers=targetLayer, use_cuda=True)
# def XGradCAM(input_tensor, cls=None):
#     targets = [ClassifierOutputTarget(cls)]
#
#     if cls is None:
#         grayscale_cam = cam(input_tensor=input_tensor, aug_smooth=True)
#     else:
#         grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True)
#     grayscale_cam = grayscale_cam[0, :]
#
#     return grayscale_cam


def closest_power_of_two(n):
    """返回数n最近的2的幂"""
    if n < 0:
        raise ValueError("n不能为负数!")
    power = 1
    while power < n:
        power *= 2
    if power - n <= n - power / 2:
        return int(power)
    else:
        return int(power / 2)

def processImg(img,SIZE=None):
    img = np.float32(img)
    if SIZE is not None:
        img = cv2.resize(img, SIZE)
    else:
        shape = [closest_power_of_two(img.shape[0]), closest_power_of_two(img.shape[1])]
        img = cv2.resize(img,shape)
    tensor = np.expand_dims(img, axis=0)
    tensor = torch.tensor(tensor).permute(0, 3, 1, 2)
    norm_img = img / 255.0
    return img,norm_img,tensor

"""
The distillation model CAMnet is invoked to generate the heatmap
"""
def modelHeat(img,model,SIZE=None,en_glcm=False,glcm_inv = False):
    img,norm_img,tensor = processImg(img,SIZE)
    tensor = tensor.to('cuda')

    hot = model(tensor)
    hot = hot.cpu().detach().numpy()
    if SIZE is None:
        hot = hot[0][0]
    else:
        hot = hot.reshape(SIZE)
    tensor.cpu()

    hot = np.where(hot > 0.5,1,hot)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    weight = np.zeros(gray.shape)

    if en_glcm:
        mean_glcm = fast_glcm.fast_glcm_mean(gray, 0, 255, 8, 5)
        weight = mean_glcm / mean_glcm.max()
        if glcm_inv:
            weight = 1 - weight

    weight = np.where(weight > 0.5, 1, weight)
    grayscale_cam = 0.5*weight + 0.5*hot
    grayscale_cam = np.where(grayscale_cam > 0.5,1,grayscale_cam)
    visualization = show_cam_on_image(norm_img, grayscale_cam, use_rgb=True)
    return visualization

"""
Call resnet101 to extract CAM to generate heat map
"""
def heatMap(input_tensor, rgb_img, frame, cls=22, en_glcm=False):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    weight = np.zeros(gray.shape)

    if en_glcm:
        mean_glcm = fast_glcm.fast_glcm_mean(gray, 0, 255, 8, 5)
        mean_glcm = mean_glcm / mean_glcm.max()
        weight += mean_glcm - 0.5

    grayscale_cam = XGradCAM(input_tensor, cls)
    grayscale_cam = 1 - grayscale_cam
    grayscale_cam = 0.8 * weight + grayscale_cam
    grayscale_cam = 1 - grayscale_cam
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return visualization


def Process(frame, ShowSize, cls=None, glcm_on=True,heat=True):
    nframe = np.float32(frame) / 255
    input_tensor = preprocess_image(nframe, mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    if heat:
        img = heatMap(input_tensor, nframe, frame, cls, glcm_on)

        img = cv2.resize(img, ShowSize)

        return img
    else:
        cam = XGradCAM(input_tensor, cls)
        return cam

def clear():
    global model
    model = model.to('cpu')
    del model
    torch.cuda.empty_cache()