# coding:utf-8

import numpy as np

a1 = np.random.random([4,2])
print(a1)
a2 = np.random.random([4,3,5])
# print(a2[...,2:4])
print('数组属性：')
print(a1.dtype)
print(a1.shape)
print(a1.size)
print(a1.itemsize) # 每个元素占用的字节数
print(a1.nbytes)

print(np.sctypeDict.keys()) # 所有numpy数据类型
print(a1.ravel())
print(a1.flatten())
print(a1)
a = np.arange(8).reshape([2,2,2])
print(a)
print(a.T)
print(np.transpose(a,axes=(1, 0, 2)))

print(np.concatenate([a, a], axis=2))

a = np.ones([3, 2])
print(np.c_[a, a])
print(np.r_[a, a])

a = np.arange(4).reshape([1, -1])
print(np.r_[a,a,a])
print(a1)
a = np.split(a1, 2,axis=0)
print(type(a))
print(len(a))