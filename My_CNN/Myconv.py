#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# File Name:        Myconv.py
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

class MyConv3x3:
    '''
    卷积类，使用3x3卷积核
    '''
    def __init__(self, num_filters):
        '''
        卷积核是一个具有 (num_filters, 3, 3) 维度的三维数组
        :param num_filters: 卷积核数:
        '''
        self.num_filters = num_filters

        # 通过除以9减小初始值的方差
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        使用有效填充生成所有可能的 3x3 图像区域
        :param image: 是一个二维数组
        '''
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def backprop(self, d_L_d_out, learn_rate):
        '''
        执行卷积层的反向传播
        :param d_L_d_out: 是该层输出的损失梯度
        :param learn_rate: 是浮点数类型
        '''
        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        # 更新卷积
        self.filters -= learn_rate * d_L_d_filters

        # 由于使用 My_Conv3x3 作为 CNN 的第一层，因此没有返回任何内容
        # 否则，需要返回这一层输入的损失梯度，就像 My_CNN 中的所有其他层一样
        return None

    def forward(self, input):
        '''
        使用给定的输入执行卷积层的前向传递
        返回具有维度 (h, w, num_filters) 的三维数组数组
        :param input: 是一个二维数组
        '''
        self.last_input = input
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        return output





