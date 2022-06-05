#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# File Name:        net.py
# Created:          2022/6/1 12:02
# Software:         PyCharm
#
# Author:           HHH
# Email:            1950049@tongji.edu.cn
# Gitee:            https://gitee.com/jin-yiyang
# Version:          v1.0
#
# Description:      Main Function: use Pytorch_CNN
#
# ------------------------------------------------------------------
# Change History :
# <Date>     | <Version> | <Author>       | <Description>
# ------------------------------------------------------------------
# 2022/6/1   | v1.0      | HHH            | Create file
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    '''
    构建pytorch网络
    '''
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=0, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size =2, stride=None, padding=0)
        self.fc1 = nn.Linear(13 * 13 * 8, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = x.view(-1, 13 * 13 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(args, model, device, train_loader, optimizer, epoch):
    '''
    训练过程
    :param args: 超参数
    :param model: 网络:
    :param device: 设备:
    :param train_loader: 训练目录:
    :param optimizer: 优化器:
    :param epoch: 训练轮数:
    :return:
    '''
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    '''
    测试过程
    :param args: 超参数
    :param model: 网络
    :param device: 设备
    :param test_loader: 测试目录
    :return:
    '''
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\n--- Testing the CNN ---')
    print('Test Loss:', test_loss)
    print('Test Accuracy:', correct / len(test_loader.dataset))



