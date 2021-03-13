# 乳腺癌数据集

> 2021.3.7

乳腺癌（breast cancer）数据集是一个经典的二分类数据集，在统计学习和机器学习领域都经常被用作示例。数据文件见同文件夹下的breast_cancer.csv。

## 数据集介绍

| 数据条数 | 569             |
| -------- | --------------- |
| 特征数   | 30              |
| 类别数   | 2               |
| 类别比例 | 212恶性+357良性 |

数据举例（前30列为特征，最后一列为类别。类别0为恶性，1为良性）：

![image-20210307144945931](http://qiniu.nkudial.top/image-20210307144945931.png)

## 数据集加载

python的scikit-learn库中集成了乳腺癌数据集的加载工具。加载方法如下：

```python
# coding: utf-8
from sklearn import datasets   # 导入scikit-learn库
breast_cancer_data = datasets.load_breast_cancer()
features = breast_cancer_data.data   # 特征
targets = breast_cancer_data.target  # 类别
```

