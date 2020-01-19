# coding:utf8
import numpy as np
from numpy import exp
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0,1,-1]])
    # print(data)
    return data[:,:2], data[:,-1]



class LogisticRegressionClassifier():

    def __init__(self, max_itr=200, alpha=0.2):
        self.max_itr = max_itr
        self.alpha = alpha

    def sigmod(self,x):
        return 1 / (1 + exp(-x))

    def fit(self,x, y):
        y = y.reshape([-1,1])
        # bias = np.ones([x.shape[0]]).reshape([-1,1])
        # x = np.c_[x,bias] # 加偏执项
        self.shape1 = x.shape[1]
        self.theta = np.random.normal(0,1,[self.shape1,1])
        # self.theta = np.array([2.,1.,-15.3]).reshape([self.shape1, 1])
        # self.theta = np.array([2.,1.,5.3]).reshape([self.shape1, 1])

        self.draw(x, y)
        for itr in range(self.max_itr):
            model = self.sigmod(np.dot(x,self.theta))
            model = model - y
            model = np.multiply(model,x)
            # model = model * x
            model = np.mean(model,axis=0)
            model = model.reshape([-1,1])
            self.theta -= self.alpha * model
            print(self.theta)
            # self.draw(x, y)

    def predict(self,x):
        # bias = np.ones([x.shape[0]]).reshape([-1,1])
        # x = np.c_[x,bias] # 加偏置项
        # self.shape1 = x.shape[1]
        proba = self.sigmod(np.dot(x, self.theta))
        return proba > 0.5

    def predict_proba(self,x):
        return self.sigmod(np.dot(x, self.theta))

    def draw(self, x, y):
        theta0, theta1, theta2 = self.theta.flatten().tolist()
        x_points = np.arange(4,8,0.01)
        y_ = -(x_points * theta0 + theta2) / theta1
        plt.scatter(x[:,0],x[:,1], c=y.reshape([-1]))
        plt.plot(x_points, y_, c='b')
        plt.show()

if __name__ == '__main__':
    print(np.random.normal(0,1,[5,1]))
    np.random.seed(1)

    x1 = np.random.normal(0,3,[2000]).reshape([-1,1])
    x2 = np.random.normal(0,3,[2000]).reshape([-1,1])
    x3 = x1 * x2
    X = np.c_[x1,x2]
    label_one_hot = []
    label = []
    for x1, x2 in X:
        if x1 > 0 and x2 > 0:
            label.append([1])
            label_one_hot.append([1, 0])
        elif x1 < 0 and x2 < 0:
            label.append([1])
            label_one_hot.append([1, 0])
        else:
            label.append([0])
            label_one_hot.append([0, 1])
    label_one_hot = np.array(label_one_hot)
    label = np.array(label).reshape([-1,1])
    print(label_one_hot)
    print(label)
    X = np.c_[X,x3]
    # x3 = X[:,0] * X[:,1]
    # X = np.c_[X,x3]

    # x_train, x_test, y_train, y_test = train_test_split(X,label, random_state=1)
    X, label = create_data()
    bias = np.ones([X.shape[0],1])
    X = np.c_[X,bias]
    x_train, x_test, y_train, y_test = train_test_split(X,label, random_state=1)
    print(x_train.shape)

    lr = LogisticRegressionClassifier()
    lr.fit(x_train,y_train)
    theta0, theta1, theta2 = lr.theta.flatten().tolist()
    result = lr.predict(x_test)
    print(result)
    print(y_test)
    score = result == y_test.reshape([-1,1])
    print(np.mean(score))

    x_points = np.arange(4,8,0.01)
    y_ = -(x_points * theta0 + theta2) / theta1

    # plt.scatter(x1,x2)
    plt.scatter(x_test[:,0],x_test[:,1], c=result.reshape([-1]))
    # plt.scatter(x_train[:,0],x_train[:,1], c=y_train.reshape([-1]))
    plt.plot(x_points,y_, c='b')
    plt.show()
