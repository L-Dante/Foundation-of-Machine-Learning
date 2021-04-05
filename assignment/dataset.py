import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()
y = df.iloc[:, 4].values
y[:50] = 1
y[50:100] = 2
y[100:150] = 3
X = df.iloc[:,[0,2]].values
plt.scatter(X[:50, 0], X[:50, 1], color='r', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='c', label='versicolor')
plt.scatter(X[-50:, 0], X[-50:, 1], color='g', label='virginica')
plt.xlabel('petal length')
plt.ylabel('sepal length')

#kmeans
k = 3
def gen_center(X_train, k):
    n_feature = X_train.shape[1]

    # 为了在数据范围内产生随机质心，首先计算各特征的统计量
    f_mean = np.mean(X_train, axis=0).reshape((1, n_feature))
    f_std = np.std(X_train, axis=0).reshape((1, n_feature))

    centers = np.random.randn(k, n_feature)*f_std+f_mean    # (k,n_feature)
    return centers
centers = gen_center(X, k)
n_sample = X.shape[0]
clusters = np.zeros(n_sample)    # 每个样本都对应一个类，相当于Y
dist = np.zeros((n_sample, k))    # 每个样本对每个质心都有一个距离

for i in range(k):
    dist[:, i] = np.linalg.norm(X-centers[i], axis=1)

clusters = np.argmin(dist, axis=1)
for i in range(k):
    centers[i] = np.mean(X[clusters == i], axis=0)
    
clusters = np.zeros(n_sample)    # 每个样本都对应一个类，相当于Y
dist = np.zeros((n_sample, k))    # 每个样本对每个质心都有一个距离


cent_pre = np.zeros(centers.shape)
cent_cur = gen_center(X,k)
cent_move = np.linalg.norm(cent_cur-cent_pre)    # 每轮迭代后质心的移动距离

epsilon = 1e-3    # 质心需要移动的最小距离
epoch = 0    # 当前迭代次数
max_iter = 50    # 最大迭代次数
while epoch < max_iter and cent_move > epsilon:
    epoch += 1
    # 首先计算每个样本离每个质心的距离
    for i in range(k):
        dist[:, i] = np.linalg.norm(X-cent_cur[i], axis=1)
    # 样本对应的类别为距离最近的质心
    clusters = np.argmin(dist, axis=1)
    cent_pre = deepcopy(cent_cur)
    # 计算每个类别下的均值坐标，更新质心
    for i in range(k):
        cent_cur[i] = np.mean(X[clusters == i], axis=0)
        plt.scatter(cent_cur[i][0], cent_cur[i][1], c='g')
    cent_move = np.linalg.norm(cent_cur-cent_pre)
    
plt.scatter(cent_cur[:, 0], cent_cur[:, 1], c='k', label='core')
plt.legend(loc='upper left')