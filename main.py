#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ------------------------------------------------------------------
# File Name:        main.py
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
# 2022/6/3   | v1.2      | HHH            | complete file
# ------------------------------------------------------------------

import mnist
import numpy as np
from My_CNN.Mymodel import MyModel
import pandas as pd

if __name__ == '__main__':
    print('MNIST CNN initialized..')
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()

    model = MyModel(8, 13 * 13 * 8, 10)

    lossplt = []
    accplt = []
    # 训练 CNN 3epoch
    for epoch in range(3):
        print('--- Epoch %d ---' % (epoch + 1))

        # 打乱训练数据
        permutation = np.random.permutation(len(train_images))
        train_images = train_images[permutation]
        train_labels = train_labels[permutation]

        # Train!
        loss = 0
        num_correct = 0
        for i, (im, label) in enumerate(zip(train_images, train_labels)):
            if i > 0 and i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, num_correct)
                )
                lossplt.append(loss/ 100)
                accplt.append(num_correct)
                loss = 0
                num_correct = 0

            l, acc = model.train(im, label)
            loss += l
            num_correct += acc

    dataframe = pd.DataFrame({'loss': lossplt, 'acc': accplt})
    dataframe.to_csv("acc_loss5.csv", index=False, sep=',')
    # Test the CNN
    print('\n--- Testing the CNN ---')
    loss = 0
    num_correct = 0
    for im, label in zip(test_images, test_labels):
        _, l, acc = model.forward(im, label)
        loss += l
        num_correct += acc

    num_tests = len(test_images)
    print('Test Loss:', loss / num_tests)
    print('Test Accuracy:', num_correct / num_tests)


