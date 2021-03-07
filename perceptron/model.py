# -*- encoding: utf-8 -*-
"""
@Description: 最简单的感知机模型：y=f(x)=sign(wx+b)。从几何上来看，相当于用一个超平面将数据点分为正类与负类。感知机模型有三个重要的要素：
1、前向过程；2、损失函数；3、梯度下降。
本文件实现最简单的感知机模型。使用误分类点到超平面的总距离作为损失函数。即 -1/||w|| \sum_{x_i \in M} y_i (wx_i+b)。简写为
-1 \sum_{x_i \in M} y_i (wx_i+b)
使用随机梯度下降法来优化模型，即任选一个超平面w, b，然后不断极小化目标函数。

算法流程如下:
1）选取初值w_0 b_0
2）在训练集中随机选取数据点(x_i, y_i)
3）如果y_i (wx_i+b) <= 0，wt <-- w_{t-1} + \yita y_i x_i   b_t <-- b_{t-1}+\yita y_i
4）转至2，直到训练集中没有误分类点，或者达到规定的训练轮数
@Time : 2021-3-7 15:26 
@File : model.py 
@Software: PyCharm
"""
import numpy as np


class Perceptron(object):
    def __init__(self, feat_dim, learning_rate):
        self.dim = feat_dim
        self.lr = learning_rate
        self.w = None
        self.b = None
        self.random_init()

    def random_init(self):
        self.w = np.random.randn(self.dim)
        self.b = np.random.randn()

    def forward(self, x):
        assert type(x) == np.ndarray, "目前只接收numpy.ndarray形式的数据"
        assert x.shape == (self.dim, ), "特征维度不同，请检查数据"
        y_pred = np.sign(self.w.dot(x) + self.b)
        return y_pred

    def backward(self, x, y_pred, y_true):
        if y_pred * y_true <= 0:
            self.w = self.w + self.lr * y_true * x
            self.b = self.b + self.lr * y_true