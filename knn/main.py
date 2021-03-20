# -*- encoding: utf-8 -*-
"""
@Description: 实现了原始的KNN，损失函数默认使用欧氏距离。
@Time : 2021-3-13 21:03 
@File : main.py 
@Software: PyCharm
"""
from model import KNN
from utils import setup_seed, dataloader
import numpy as np
from sklearn.metrics import accuracy_score


seed = 1234
setup_seed(seed)
x_train, x_test, y_train, y_test = dataloader(ratio=0.9)
knn = KNN(k=5)
test_num = x_test.shape[0]

# train
knn.train(instences=x_train, labels=y_train)
# test
y_pred_list = list()
for instance_id in range(test_num):
    x = x_test[instance_id]
    y_pred = knn.eval(x)
    y_pred_list.append(y_pred)

y_pred_array = np.array(y_pred_list)
print("Final Accuracy: {}%.".format(accuracy_score(y_pred_array, y_test) * 100))
