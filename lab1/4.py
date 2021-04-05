# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

C = np.array([[2.0,1.0], [1.0,2]])
A = np.linalg.cholesky(C)
X = np.random.randn(1000, 2)
Y = X @ A.T
theta = np.pi/3
u = [np.sin(theta), np.cos(theta)]
yp = Y @ u
nPoints = 50
pVars = np.zeros(nPoints)
thRange = np.linspace(0, 2*np.pi, nPoints)
for n in range(nPoints):
    theta = thRange[n]
    u = [np.sin(theta), np.cos(theta)]
    pVars[n] = np.var(Y @ u)

fig, ax = plt.subplots(figsize=(5,3))
ax.plot(pVars)
value, vector = np.linalg.eig(C)
print(value)
print(vector)