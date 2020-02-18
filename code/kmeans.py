import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
np.random.seed(0)
class Kmeans(object):

    def __init__(self, k, max_itr=300):
        self.k = k
        self.max_itr = max_itr

    def fit(self, x, style='Kmeans', threshold=1e-10):
        labels = np.zeros(len(x), dtype=int)
        centers = x[np.random.choice(len(x), self.k)]
        old_centers = x[np.random.choice(len(x), self.k)]

        observer(0, labels, centers)
        for i in range(self.max_itr):
            dist = self.calc_dist(x, centers)
            labels = dist.argmin(axis=1)
            observer(i, labels, centers)
            if style == 'Kmeans':
                for i in range(self.k):
                    idx = labels == i
                    centers[i] = np.mean(x[idx],axis=0)
            elif style == 'Kmedoids':
                for i in range(self.k):
                    idx = labels == i
                    dist = self.calc_dist(x[idx], x[idx])
                    dist = np.sum(dist, axis=1)
                    center = np.argmin(dist)
                    centers[i] = x[idx][center]
                pass

            if self.delta(centers, old_centers) < threshold:
                return x, centers, labels,
            else:
                old_centers = centers.copy()

        pass

    def delta(self, x, y):
        return np.sum((x-y)**2)

    def calc_dist(self, x, y):
        batch_size = len(x)
        num_class = len(y)
        xx = np.sum(x * x, axis=1)
        yy = np.sum(y * y, axis=1)
        xy = np.dot(x,y.T)
        return np.tile(xx,(num_class, 1)).T + np.tile(yy,(batch_size,1)) - 2*xy

if __name__ == '__main__':
    X, _ = make_blobs(n_samples=1500, random_state=170)
    def observer(iter, labels, centers):
        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

        # 绘制数据点
        data_colors = [colors[lbl] for lbl in labels]
        plt.scatter(X[:, 0], X[:, 1], c=labels, s=100, alpha=0.2, marker="o")
        # 绘制中心
        plt.scatter(centers[:, 0], centers[:, 1], s=200, c='k', marker="^")

        plt.show()
    pass

    print(X)
    model = Kmeans(3,)
    model.fit(X, style='Kmedoids')