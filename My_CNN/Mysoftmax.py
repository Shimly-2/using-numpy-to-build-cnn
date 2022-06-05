#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# File Name:        Mysoftmax.py
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

class MySoftmax:
    '''
    具有 softmax 激活的标准全连接层
    '''
    def __init__(self, input_len, nodes):
        '''
        除以 input_len 以减少初始值的方差
        :param input_len: 输入图像shape:
        :param nodes: 输出节点数:
        '''
        self.weights = np.random.randn(input_len, nodes) / input_len
        self.biases = np.zeros(nodes)

    def backprop(self, d_L_d_out, learn_rate):
        '''
        执行 softmax 层的反向传播
        :return: 该层输入的损失梯度
        :param d_L_d_out: 是该层输出的损失梯度
        '''
        # d_L_d_out 只有 1 个元素是非零的
        for i, gradient in enumerate(d_L_d_out):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # out[i] 对总数的梯度
            d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
            d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # 针对weights/biases/input的总计梯度
            d_t_d_w = self.last_input
            d_t_d_b = 1
            d_t_d_inputs = self.weights

            # 总损失梯度
            d_L_d_t = gradient * d_out_d_t

            # weights/biases/input的损失梯度
            d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
            d_L_d_b = d_L_d_t * d_t_d_b
            d_L_d_inputs = d_t_d_inputs @ d_L_d_t

            # 更新 weights/biases
            self.weights -= learn_rate * d_L_d_w
            self.biases -= learn_rate * d_L_d_b
            return d_L_d_inputs.reshape(self.last_input_shape)

    def forward(self, input):
        '''
        使用给定的输入执行 softmax 层的前向传递
        :return: 包含相应概率值的一维数组。
        :param input: 可以是任意维度的任意数组
        '''
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.weights.shape

        totals = np.dot(input, self.weights) + self.biases
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)