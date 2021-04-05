# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 21:06:16 2020

@author: 12605
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

NumDataPerClass = 200

m1 = [[0,5]]
m2 = [[5,0]]
C = [[2, 1], [1, 2]]

A = np.linalg.cholesky(C)

U1 = np.random.randn(NumDataPerClass, 2)
X1 = U1 @ A.T + m1

U2 = np.random.randn(NumDataPerClass, 2)
X2 = U2 @ A.T + m2

X = np.concatenate((X1, X2), axis=0)

labelPos = np.ones(NumDataPerClass)
labelNeg = -1 * np.ones(NumDataPerClass)
y = np.concatenate((labelPos, labelNeg))

rIndex = np.random.permutation(2*NumDataPerClass)
Xr = X[rIndex]
yr = y[rIndex]

X_train = Xr[0:NumDataPerClass]
y_train = yr[0:NumDataPerClass]
X_test = Xr[NumDataPerClass:2*NumDataPerClass]
y_test = yr[NumDataPerClass:2*NumDataPerClass]
Ntrain = NumDataPerClass
Ntest = NumDataPerClass

def PercentCorrect(Inputs, targets, weights):
    N = len(targets)
    nCorrect = 0
    for n in range(N):
        OneInput = Inputs[n,:]
        if (targets[n] * np.dot(OneInput, weights) > 0):
            nCorrect +=1
    return 100*nCorrect/N

w = np.random.randn(2)
print(w)
print('Initial Percentage Correct: %6.5f' %(PercentCorrect(X_train, y_train, w)))

MaxIter=1000

alpha = 0.002

P_train = np.zeros(MaxIter)
P_test = np.zeros(MaxIter)

for iter in range(MaxIter):

    r = np.floor(np.random.rand()*Ntrain).astype(int)
    x = X_train[r,:]

    if (y_train[r] * np.dot(x, w) < 0):
        w += alpha * y_train[r] * x

    P_train[iter] = PercentCorrect(X_train, y_train, w);
    P_test[iter] = PercentCorrect(X_test, y_test, w);
    
print('Percentage Correct After My Training: %6.2f %6.2f'
%(PercentCorrect(X_train, y_train, w), PercentCorrect(X_test, y_test, w)))


plt.scatter(X[:,0], X[:,1])
plt.plot([0,-1*w[0]*10], [0,w[1]*10])
plt.show()
