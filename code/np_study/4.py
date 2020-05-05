# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

a = np.array([[1,2,1],[2,3,4],[2,2,2]])
a1 = np.mat(a)
print(np.linalg.inv(a))
print(a1.I)