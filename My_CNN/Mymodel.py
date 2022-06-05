#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# File Name:        Mymodel.py
# Created:          2022/6/1 12:02
# Software:         PyCharm
#
# Author:           HHH
# Email:            1950049@tongji.edu.cn
# Gitee:            https://gitee.com/jin-yiyang
# Version:          v1.0
#
# Description:      Main Function: use My_CNN
#
# ------------------------------------------------------------------
# Change History :
# <Date>     | <Version> | <Author>       | <Description>
# ------------------------------------------------------------------
# 2022/6/1   | v1.0      | HHH            | Create file
# ------------------------------------------------------------------

import numpy as np
from My_CNN.Myconv import MyConv3x3
from My_CNN.Mymaxpool import MyMaxPool2
from My_CNN.Mysoftmax import MySoftmax

class MyModel:
    '''
    我的CNN模型实例
    '''
    def __init__(self, num_filters, input_len, nodes):
        '''
        初始化我的CNN模型参数
        :param num_filters: 卷积核数量:
        :param input_len: 输入参数shape:
        :param nodes: 输出节点数:
        '''
        self.conv = MyConv3x3(num_filters)          # 28x28x1 -> 26x26x8
        self.pool = MyMaxPool2()                    # 26x26x8 -> 13x13x8
        self.softmax = MySoftmax(input_len, nodes)  # 13x13x8 -> 10

    def forward(self, image, label):
        '''
        完成 CNN 的前向传递并计算准确度和交叉熵损失
        :param image: 是一个二维数组
        :param label: 是一个数字
        '''
        # 将图像从 [0, 255] 转换为 [-0.5, 0.5] 以使其更易于使用
        # 属于数据集的标准化
        out = self.conv.forward((image / 255) - 0.5)
        out = self.pool.forward(out)
        out = self.softmax.forward(out)

        # 计算交叉熵损失和准确度， np.log() 是自然对数。
        loss = -np.log(out[label])
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc

    def train(self, image, label, lr=.005):
        '''
       完成给定图像和标签的完整训练步骤
        :return: 交叉熵损失和准确性
        :param image: 是一个二维数组
        :param label: 是一个数字
        :param lr: 是学习率
        '''
        # 前向传递
        out, loss, acc = self.forward(image, label)

        # 计算初始梯度
        gradient = np.zeros(10)
        gradient[label] = -1 / out[label]

        # 反向传播
        gradient = self.softmax.backprop(gradient, lr)
        gradient = self.pool.backprop(gradient)
        gradient = self.conv.backprop(gradient, lr)

        return loss, acc