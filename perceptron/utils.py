# -*- encoding: utf-8 -*-
"""
@Description: 使用到的工具。比如感知机要求类别为-1和1，所以要将0类改成-1类。
@Time : 2021-3-7 15:26 
@File : utils.py 
@Software: PyCharm
"""
from sklearn import datasets
from sklearn import model_selection
import numpy as np
import random


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def relabel(origin_y):
    y_temp = origin_y - 1
    final_y = y_temp + origin_y
    return final_y


def dataloader(ratio=0.9):   # ratio trainset:allset
    breast_cancer_data = datasets.load_breast_cancer()
    features = breast_cancer_data.data   # 特征
    targets = breast_cancer_data.target  # 类别
    assert len(set(targets)) == 2, "目前我们只解决二分类问题"
    assert targets.max() == 1, "应该至少保证有一类为1"
    if targets.min() == 0:
        targets = relabel(targets)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, targets, train_size=ratio)
    return x_train, x_test, y_train, y_test