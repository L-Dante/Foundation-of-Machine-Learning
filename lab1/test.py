import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 3)
y = np.linspace(0, 1, 2)
xx,yy = np.meshgrid(x, y, indexing = 'ij')
