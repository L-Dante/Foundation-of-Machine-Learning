# -*- coding: utf-8 -*-
import numpy as np

realUtility = np.loadtxt("D:\document\javaworkspace\genetic\GrealUtility.txt")
predictUtility = np.loadtxt("D:\document\javaworkspace\genetic\GpreUtility.txt")
molecular = 0.0
denominator1 = 0.0
denominator2 = 0.0
denominator = 0.0
for i in range(len(realUtility)):
    molecular += (realUtility[i]-realUtility.mean()) * (predictUtility[i]-predictUtility.mean())
    denominator1 += np.square((realUtility[i]-realUtility.mean()))
    denominator2 += np.square((predictUtility[i]-predictUtility.mean()))
    
denominator = np.sqrt(denominator1*denominator2)
person = molecular/denominator
print(person)                              