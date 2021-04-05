from sklearn.datasets import make_spd_matrix
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def genGaussianSamples(N, m, C):
    A = np.linalg.cholesky(C)
    U = np.random.randn(N,2)
    return(U @ A.T + m)

def gauss_2D_value(x,m,c):
    c_inv=np.linalg.inv(c)
    c_det=np.linalg.det(c)
    num=np.exp(-0.5*np.dot((x-m).T,np.dot(c_inv,(x-m))))
    den=2*np.pi*np.sqrt(c_det)
    return num/den

def gauss_2D_grid_values(n,m,c,xlim,ylim):
    x=np.linspace(xlim[0],xlim[1],n)
    y=np.linspace(ylim[0],ylim[1],n)
    X,Y=np.meshgrid(x,y,indexing='ij')
    Z=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            xvec=np.array([X[i,j],Y[i,j]])
            Z[i,j]=gauss_2D_value(xvec,m,c)
    return X,Y,Z

def plot_contour(m,c,xlim,ylim,n=50,level=5,cmap=None):
    grid_x,grid_y,grid_z=gauss_2D_grid_values(n,m,c,xlim,ylim)
    plt.contour(grid_x,grid_y,grid_z,level,cmap=cmap)

# Define three means
#
Means = np.array([[0, 3], [3, 0], [4,4]])
# Define three covariance matrices ensuring
# they are positive definite
#

CovMatrices = np.zeros((3,2,2))
for j in range(3):
    CovMatrices[j,:,:] = make_spd_matrix(2)


# Priors
#
k = 3
w = np.random.rand(k)
w = w / np.sum(w)
# How many data in each component (1000 in total)
#
nData = np.floor(w * 1000).astype(int)
# Draw samples from each component
#
X0 = genGaussianSamples(nData[0], Means[0,:], CovMatrices[0,:,:])
X1 = genGaussianSamples(nData[1], Means[1,:], CovMatrices[1,:,:])
X2 = genGaussianSamples(nData[2], Means[2,:], CovMatrices[2,:,:])
# Append into an array for the data we need
#
X = np.append(np.append(X0, X1, axis=0), X2, axis=0)

#draw scatter
xlim = [-4,8]
ylim = [-2,6]
#plt.scatter(X0[:,0], X0[:,1], c = 'r', alpha=0.5, marker = 'x')
#plt.scatter(X1[:,0], X1[:,1], c = 'y', alpha=0.5, marker = 'x')
#plt.scatter(X2[:,0], X2[:,1], c = 'b', alpha=0.5, marker = 'x')
plt.scatter(X[:,0], X[:,1], c = 'r', alpha=0.5, marker = 'x')
#plot_contour(Means[0], CovMatrices[0], xlim, ylim)
#plot_contour(Means[1], CovMatrices[1], xlim, ylim)
#plot_contour(Means[2], CovMatrices[2], xlim, ylim)

#kmeans
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
    
plt.scatter(cent_cur[:, 0], cent_cur[:, 1], c='k')

