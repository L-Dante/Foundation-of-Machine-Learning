# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import preprocessing
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from matplotlib.pyplot import boxplot
from sklearn.linear_model import LinearRegression
from sklearn import svm

diabetes = datasets.load_diabetes()
X = diabetes.data
# normalize the input data
X = preprocessing.scale(X)
y = diabetes.target

def gaussian(x, u, sigma):
    return(np.exp(-0.5 * np.linalg.norm(x-u) / sigma))

def error(t, t_hat):
    return ((t - t_hat)**2).mean()

N, p = X.shape
print(N, p)

# cross validation
kf = KFold(n_splits=10)
f = 1
l_training_error = []
l_test_error = []
for train_idx, test_idx in kf.split(X, y):
# spit data equally into training and test test sets.
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    N_train, p_train = len(X_train), len(y_train)
    N_test, p_test = len(X_test), len(y_test)
    
    M = 200
    U_train = np.zeros((N_train, M))
    U_test = np.zeros((N_test, M))
    
    C = np.random.randn(M, p)
    
    x1 = X[0,:]
    x2 = X[2,:]
    # calculate the the average of several pairwise distances.
    distance = pdist(X,metric='euclidean')
    sigma = np.mean(distance)
    
    # use K-means to  set the basis function locations to the cluster centres.
    estimator = KMeans(n_clusters=M)
    estimator.fit(X_train)
    centroids = estimator.cluster_centers_ 
    
    # training
    for i in range(N_train):
        for j in range(M):
            U_train[i,j] = gaussian(X_train[i,:], centroids[j,:], sigma)
            
    l_train = np.linalg.inv(U_train.T @ U_train) @ U_train.T @ y_train
    
    # test
    for i in range(N_test):
        for j in range(M):
            U_test[i,j] = gaussian(X_test[i,:], centroids[j,:], sigma)
    
    yh_train = U_train @ l_train
    yh_test = U_test @ l_train
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(9, 3))
    ax[0].boxplot(yh_test)
    ax[0].set_title('RBF')
    
    # LinearRegression
    '''
    LR = LinearRegression()
    LR.fit(X_train, y_train)
    ax[1].boxplot(LR.predict(X_test))
    ax[1].set_title('LinearRegression')
    '''
    
    # sklearn inbuilt rbf model
    classifier_model = svm.SVC(C = 1.0, kernel = 'rbf', degree = 3, gamma = 'auto')
    classifier_model.fit(X_train, y_train)
    ax[1].boxplot(classifier_model.predict(X_test))
    ax[1].set_title('sklearn rbf')
    
    '''
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
    ax[0].scatter(y_train, yh_train, c='m', s=3)
    ax[0].grid(True)
    ax[0].set_title("Training Set", fontsize=14)
    ax[0].set_xlabel("True Target", fontsize=12)
    ax[0].set_ylabel("Prediction", fontsize=12)
    
    ax[1].scatter(y_test, yh_test, c='y', s=3)
    ax[1].grid(True)
    ax[1].set_title("Test Set", fontsize=14)
    ax[1].set_xlabel("True Target", fontsize=12)
    ax[1].set_ylabel("Prediction", fontsize=12)
    
    train_err = error(y_train, yh_train)
    test_err = error(y_test, yh_test)
    l_training_error.append(train_err)
    l_test_error.append(test_err)
    print("loop %d", f)
    print(train_err)
    print(test_err)
    f += 1
    '''


