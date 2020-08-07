# coding:utf-8

import numpy as np

I = np.eye(3)
print(I)
np.savetxt('eye.txt', I)

a = np.random.random([4,2])
print(a)
print(np.median(a, axis=0))
print(a.var(axis=1))
print(a.std(axis=1))
print(np.diff(-a, axis=0))
indices = np.where(a > 0.5)
print(np.take(a, indices))