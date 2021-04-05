import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

DataNum = 100
NumDataPerClass = math.ceil(DataNum/2)
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.tail()
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
#O = np.ones((2*NumDataPerClass, 1))
#X = np.append(X, O, axis=1)

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

#draw
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')

plt.show()

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(range(MaxIter), P_train, 'b', label = "Training")
ax.plot(range(MaxIter), P_test, 'r', label = "Test")
ax.grid(True)
ax.legend()
ax.set_title('Perceptron Learning(alpha=0.002)')
ax.set_ylabel('Training and Test Accuracies', fontsize=14)
ax.set_xlabel('Iteration', fontsize=14)
plt.savefig('learningCurves.png')
