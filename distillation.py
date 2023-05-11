#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2023/4/26 12:27
# !@Author  : murInj
# !@Filer    : .py

import torch
import torch.nn as nn
import torch.optim as optim
from mobileNet import MobileNetV3_Small
from torch.nn import MSELoss
import dataset

"""
This file distills CAM weights from resnet101 into CAMnet
"""

"""PARAMS"""
BATCH_SIZE = 5
INPUT_SIZE = (BATCH_SIZE, 320, 320, 3)

LOAD = False
model = None

if __name__ == '__main__':
    if LOAD:
        model = torch.load("mobile.pth")
    else:
        model = MobileNetV3_Small(act=nn.ReLU)

    dataset = dataset.camDataset("../CAM/img", INPUT_SIZE,load=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


    criterion = MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epoch = 0
    best_model = None
    min_loss = 0x3f3f3f
    while True:
        model.train()

        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.permute(0, 3, 1, 2)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            inputs = inputs.to('cpu')
            labels = labels.to('cpu')

        epoch_loss = running_loss / len(dataset)
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            best_model = model
            if epoch != 0:
                torch.save(best_model, "mobile.pth")
                print("model update:{}".format(min_loss))

        print('epoch: {} Loss: {:.4f}'.format(epoch, epoch_loss))
        epoch += 1
