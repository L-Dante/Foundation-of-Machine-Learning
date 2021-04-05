# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def gauss_2D_sample(m,c,n=200):
    return np.random.multivariate_normal(m,c,n)

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

def gauss_2D_posterior_value(x,m1,c1,p1,m2,c2,p2):
    num=p1*gauss_2D_value(x,m1,c1)
    den=p1*gauss_2D_value(x,m1,c1)+p2*gauss_2D_value(x,m2,c2)
    return num/den

def gauss_2D_posterior_grid_values(n,m1,c1,p1,m2,c2,p2,xlim,ylim):
    x=np.linspace(xlim[0],xlim[1],n)
    y=np.linspace(ylim[0],ylim[1],n)
    X,Y=np.meshgrid(x,y,indexing='ij')
    Z=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            xvec=np.array([X[i,j],Y[i,j]])
            Z[i,j]=gauss_2D_posterior_value(xvec,m1,c1, p1, m2,c2,p2 )
    return X,Y,Z

def plot_sample(m,c,axis,ps=None):
    if ps is None:
        ps=gauss_2D_sample(m,c)
    axis.scatter([p[0] for p in ps],[p[1] for p in ps], alpha=0.5)
    
def plot_contour(m,c,xlim,ylim,axis,n=50,level=5,cmap=None):
    grid_x,grid_y,grid_z=gauss_2D_grid_values(n,m,c,xlim,ylim )
    axis.contour(grid_x,grid_y,grid_z,level,cmap=cmap)
    
def plot_posterior(m1,c1,p1,m2,c2,p2,xlim,ylim,axis,n=50,level=5,cmap=None):
    grid_x,grid_y,grid_z=gauss_2D_posterior_grid_values(n,m1,c1,p1,m2,c2,p2,xlim,ylim)
    axis.contour(grid_x,grid_y,grid_z,level,cmap=cmap)
    
m1 = [0,3]
m1 = np.asarray(m1)
c1 = [[2,1],[1,2]]
c1 = np.asarray(c1)

m2 = [3,2.5]
m2 = np.asarray(m2)
c2 = [[2,1],[1,2]]
c2 = np.asarray(c2)

xlim = [-5,8]
ylim = [-4,9]

def fisher_lda(m1, m2, c1, c2):
    c = c1 + c2
    c_inv = np.linalg.inv(c)
    return c_inv @ (m1 - m2)

def project(w, ps):
    return ps @ w

ps1 = gauss_2D_sample(m1, c1)
ps2 = gauss_2D_sample(m2, c2)

wf = fisher_lda(m1, m2, c1, c2)

proj1 = project(wf, ps1)
proj2 = project(wf, ps2)

def calculate_roc(yp1,yp2,n=50):
    pmin = np.min( np.array( (np.min(yp1), np.min(yp2) )))
    pmax = np.max( np.array( (np.max(yp1), np.max(yp2) )))
    thRange = np.linspace(pmin, pmax, n)
    ROC = np.zeros((n, 2))
    for i, thresh in enumerate(thRange):
        TP = len(yp2[yp2 > thresh])  / len(yp2)
        FP = len(yp1[yp1 > thresh])  / len(yp1)
        ROC[i,:] = [TP, FP]
    return ROC, thRange

def calculate_auc(roc):
    return np.trapz(roc[:,1][::-1], roc[:, 0][::-1])

def accuracy_at_threshold(thresh):
    tp = len(proj1[proj1 >= thresh])
    tn = len(proj2[proj2 <= thresh])
    acc = (tp + tn)/(len(proj1) + len(proj2))
    return acc

def find_best_thresh(roc, thresh_range):
    dist_to_best = []
    for p in roc:
        d = np.linalg.norm(p - np.asarray([0,1]))
        dist_to_best.append(d)
    best_thresh = thresh_range[np.argmin(dist_to_best)]
    best_acc = accuracy_at_threshold(best_thresh)
    return best_thresh, best_acc

def plot_roc(roc, axis, label=None, color=None):
    axis.plot(roc[:,0],roc[:,1],label=label, color=color)
    axis.set_xlabel('False Positive Rate')
    axis.set_ylabel('True Positive Rate')
    axis.set_title('ROC')
    axis.grid(True)

roc, thresh_range = calculate_roc(proj1, proj2)
auc = calculate_auc(roc)
best_thresh, best_acc = find_best_thresh(roc, thresh_range)

fig, axis = plt.subplots(figsize=(6,6))
plot_roc(roc, axis)

w_rand = np.random.uniform(size=(2,))
w_mean = m1 - m2

proj_rand1 = project(w_rand, ps1)
proj_rand2 = project(w_rand, ps2)
roc_rand, _ = calculate_roc(proj_rand1, proj_rand2)
auc_rand = calculate_auc(roc_rand)

proj_mean1 = project(w_mean, ps1)
proj_mean2 = project(w_mean, ps2)
roc_mean, _ = calculate_roc(proj_mean1, proj_mean2)
auc_mean = calculate_auc(roc_mean)

print("auc: ", auc)
print("auc_rand: ", auc_rand)
print("auc_mean: ", auc_mean)

fig, axes = plt.subplots()
plot_roc(roc, axes, color='m', label='Fisher')
plot_roc(roc_rand, axes, label='Random')
plot_roc(roc_mean, axes, label='Mean Intersection')
axes.legend(loc='best')

