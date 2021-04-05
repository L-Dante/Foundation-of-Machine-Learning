# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


flag = 1
data = np.loadtxt("D:\document\javaworkspace\group6\distance_2000.txt")


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
datasize = 1999

x = np.linspace(0, datasize, num = datasize)
#plt.scatter(x, PreUtil, label="PreUtil")
#plt.scatter(x, RealUtil, label="RealUtil")
#plt.legend()
#plt.savefig("UtilityCompare.jpg")

Distance = []
for i in range(datasize):
    Distance.append(data[i])


plt.scatter(x, Distance, label="distance")
plt.legend()
plt.title('distance in domain size 2000')
ax.set_ylabel('distance')
plt.savefig("distance_2000.jpg")
