# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#df= pd.read_csv('data-01')
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
print(y)