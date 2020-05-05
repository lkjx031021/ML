# coding:utf-8

import numpy as np
import matplotlib.pyplot as plt


a1 = np.array([1,2,3])
a2 = np.array([-5,1,1])
a3 = np.array([9,2,3])
print(np.maximum(a1,a2))

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
# plt.plot(x, y)
# plt.show()

a = np.array([1.,2,3])
b = np.array([5.,2,3])
np.add(a,b,out=a)
print(a)

w = np.array([1.,0.5,2.])
a = np.arange(10)
print(a)
print(np.convolve(w, a, mode='valid'))
print(np.convolve(w, a, mode='full'))

a = np.array([1,2,3])
b = np.array([3.2,2,1])
print(np.cov(a, b))
print(np.cov(a, b).diagonal()) # 对角线元素
print(np.cov(a, b).trace()) # 对角线元素之和：迹
print(np.corrcoef(a, b))    # 相关系数
# print(np.polyfit())

a = np.array([0,1,2,3,4,0,0])
print(np.trim_zeros(a))
w1 = np.hanning(5)
print(w1)
