# -*- encoding: utf-8 -*-
"""
@Description: 
@Time : 2021-3-13 21:03 
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


def dataloader(ratio=0.9):   # ratio trainset:allset
    breast_cancer_data = datasets.load_breast_cancer()
    features = breast_cancer_data.data   # 特征
    targets = breast_cancer_data.target  # 类别
    assert len(set(targets)) == 2, "目前我们只解决二分类问题"
    features = feature_scaler(features)
    assert (features.max() == 1) and (features.min() == 0), "特征放缩失败"
    x_train, x_test, y_train, y_test = model_selection.train_test_split(features, targets, train_size=ratio)
    return x_train, x_test, y_train, y_test


def find_majority(labels):
    """
    找到k个邻居中个数最多的标签并返回，若有多个相同个数的标签，则返回第一个。
    :param labels: lens = k，标签的列表
    :return: k个邻居中个数最多的标签
    """
    label_freq_dict = dict()   # key--label, value--frequency
    for label in labels:
        if label in label_freq_dict:
            label_freq_dict[label] += 1
        else:
            label_freq_dict[label] = 1

    majority_label = max(label_freq_dict.items(), key=lambda x: x[1])[0]
    return majority_label


def feature_scaler(features):
    """
    knn需要计算距离，而数据集中不同的特征尺度是不同的，因此我们先将所有特征缩放至相同的尺度，以防尺度的差异影响最终分类性能
    好像其实没啥用，2021.3.18
    :param features: 特征
    :return: 归到[0, 1]后的特征
    """
    feature_max = features.max(axis=0)
    feature_min = features.min(axis=0)
    scaled_features = (features - feature_min) / (feature_max - feature_min)
    return scaled_features
