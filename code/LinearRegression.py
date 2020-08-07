import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def model(x, a, b, c):
    return a*x**2 + b*x + c

def gradf(x, d, a, b, c):
    y = model(x, a, b, c)
    grada = 2 * (y - d) * x**2
    gradb = 2 * (y - d) * x
    gradc = 2 * (y - d)
    return [grada, gradb, gradc]


x = np.random.normal(1,1,[1000])
d = x ** 2 + 2 * x + np.random.normal(0,0.4,[1000])

a, b, c = 0, 0, 1    # 初始值
batch_size = 10      # 每次训练样本的个数
eta = 0.01            # 学习率

for itr in range(1000):
    idx = np.random.randint(0, 1000, batch_size)
    xin = x[idx]
    din = d[idx]
    ga, gb, gc = gradf(xin,din,a,b,c)
    ga = np.mean(ga)
    gb = np.mean(gb)
    gc = np.mean(gc)
    a = a - ga * eta
    b = b - gb * eta
    c = c - gc * eta
    # if _b > 1:
    #     _b = 1
    # b = _b
    print(a,b,c)


xplt = np.linspace(-3, 5, 1000)
ypred = model(xplt, a, b, c)

plt.scatter(x, d)
# plt.plot(xplt, ypred, lw=2)
plt.show()

# 预测过程
import matplotlib
matplotlib.rcParams['font.sans-serif'] = 'SimHei'
# 输入x
X = np.linspace(-2, 4, 100)
# 输出y
y = model(X, a, b, c)
# 绘图
# plt.scatter(x[:, 0], d[:, 0], s=20, alpha=0.4, label="数据散点")
plt.scatter(x, d, s=20, alpha=0.4, label="数据散点")
plt.plot(X, y, lw=5, color="#990000", alpha=0.5, label="预测关系")
plt.legend()
plt.show()

print(r2_score(d, model(x, a, b, c)))