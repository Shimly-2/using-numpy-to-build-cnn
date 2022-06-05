#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# File Name:        Mymaxpool.py
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

class MyMaxPool2:
    '''
    使用池大小为 2 的 最大池化层
    '''
    def iterate_regions(self, image):
        '''
        生成非重叠的 2x2 图像区域以进行池化
        :param image: 是一个二维数组
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j

    def backprop(self, d_L_d_out):
        '''
        执行最大池化层的反向传播
        :return: 此层输入的损失梯度
        :param d_L_d_out: 是该层输出的损失梯度
        '''
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # 如果这个像素是最大值，将损失梯度复制到它
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

    def forward(self, input):
        '''
        使用给定的输入执行最大池化层的前向传递
        :return: 具有维度（h / 2、w / 2、num_filters）的三维数组
        :param input: 是一个具有维度 (h, w, num_filters) 的三维数组
        '''
        self.last_input = input
        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output