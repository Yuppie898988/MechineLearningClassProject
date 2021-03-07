# -*- encoding: utf-8 -*-
"""
@Description: 
@Time : 2021-3-7 15:26 
@File : main.py 
@Software: PyCharm
"""
from model import Perceptron
from utils import setup_seed, dataloader
import random
import numpy as np
from sklearn.metrics import accuracy_score


# hyperparameter
iters = 50000
lr = 0.0001
seed = 1234

# main process
setup_seed(seed)
x_train, x_test, y_train, y_test = dataloader(ratio=0.9)
perceptron = Perceptron(feat_dim=30, learning_rate=lr)
train_num = x_train.shape[0]
test_num = x_test.shape[0]

# train
for iter in range(iters):  # 每次随机从训练集中选择一个样本进行训练
    instance_id = random.randint(0, train_num - 1)
    x = x_train[instance_id]
    y_true = y_train[instance_id]
    y_pred = perceptron.forward(x)
    perceptron.backward(x, y_pred=y_pred, y_true=y_true)
    print("iteration {} finished. y_true: {}, y_pred: {}.".format(iter, y_true, y_pred))

# test
y_pred_list = list()
for instance_id in range(test_num):
    x = x_test[instance_id]
    y_pred = perceptron.forward(x)
    y_pred_list.append(y_pred)

# evaluate
y_pred_array = np.array(y_pred_list)
print("Final Accuracy: {}%.".format(accuracy_score(y_pred_array, y_test) * 100))