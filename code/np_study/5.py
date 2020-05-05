# coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from mpl_toolkits.mplot3d import Axes3D
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def model(x):
    return x ** 3

def grad(x):
    return 3 * x ** 2

x = np.linspace(-1,1,100)
y = model(x)
d = grad(x)

# plt.subplot(141)
# plt.plot(x, y)
# plt.plot(x, d, 'g--')
# plt.subplot(142)
# plt.plot(x, y)
# plt.plot(x, d, 'b--')
# plt.subplot(143)
# plt.plot(x, y)
# plt.plot(x, d, 'r--')
# plt.xlabel('shouyi')
# plt.ylabel('value')
# plt.title('你猜猜')
# plt.subplot(144)
# plt.plot(x, y, label='sazi')
# plt.plot(x, d, 'k--', lw='1', label='auto')
# plt.plot(x, d * 4, c='c', ls='-.', lw='1', label='autop')
# plt.legend()
# plt.show()
# plt.cla()

# x = np.arange(5)
# y = x * 2
# plt.hist(y)
# plt.show()

data = load_iris()
x = data['data']
y = data['target']
print(x.shape)
print(x[1])

plt.scatter(x[:,0], x[:, 1],c=y, s=np.arange(150))
plt.scatter(4.9, 3.,marker='^',lw=5,c='g')
plt.grid()
plt.show()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x[:,0], x[:, 1], x[:, 2], c=y)
plt.show()

