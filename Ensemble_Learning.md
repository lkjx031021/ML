## XGBoost

​		GBoost是boosting算法的其中一种。Boosting算法的思想是将许多弱分类器集成在一起形成一个强分类器。因为XGBoost是一种提升树模型，所以它是将许多树模型集成在一起，形成一个很强的分类器。而所用到的树模型则是CART回归树模型。

​		该算法思想是就是不断得添加树，不断地进行特征分裂来生长一棵树，每次添加一棵树，其实就是学习一个新的函数，去拟合上次预测的残差，当我们训练完成得到K棵树，我要预测一个样本的分数，就是根据这个样本的特征，在每棵树种回落到对应的一个叶子节点，每个叶子节点，就对应一个分数，即预测值，最后只需要每棵树对应的分数加起来就是该样本的预测值。

### 2.考虑使用二阶导信息

目标函数：$J(f_t)=\sum_{i=1}^nL(y_i,\hat y_i^{t-1}+f_t(x_i))+\Omega(f_t)+C$

根据泰勒展开式：

$f(x+\Delta x)\approx f(x) + f^\prime (x)\Delta x+\frac{1}{2}f^{\prime \prime}\Delta x^2$

令，
$$
g_i=\frac{\partial L(y_i,\hat y_i^{(t-1)})}{\partial\hat y_i^{t-1}}，h_i=\frac{\partial ^2 L(y_i,\hat y_i^{(t-1)})}{\partial\hat y_i^{t-1}}
$$
得：
$$
J(f_t) \approx\sum_{i=1}^{n}\left[L(y_i,\hat y_i^{(t-1)})+g_if_t(x_i) +\frac{1}{2}h_if_t^2(x_i)\right] + \Omega(f_t)+C
$$




### 3.目标函数计算

$$
\begin{align}
J(f_t) &=\sum_{i=1}^{n}\left[L(y_i,\hat y_i^{(t-1)})+g_if_t(x_i) +\frac{1}{2}h_if_t^2(x_i)\right] + \Omega(f_t)+C \\
&=\sum_{i=1}^{n}\left[g_if_t(x_i) +\frac{1}{2}h_if_t^2(x_i)\right] + \Omega(f_t)+C\\
&=\sum_{i=1}^{n}\left[g_iw_{q(x_i)} +\frac{1}{2}h_iw_{q(x_i)}^2\right] + \gamma \cdot T + \lambda\cdot \frac{1}{2}\sum_{j=1}^Tw_j^2+C\\
&=\sum_{j=1}^T\left[\left(\sum_{i\in I_j}g_i\right) w_j +\frac{1}{2}\left(\sum_{i\in I_j}h_i\right)w_j^2 \right] + \gamma \cdot T + \lambda\cdot \frac{1}{2}\sum_{j=1}^Tw_j^2+C \\
&=\sum_{j=1}^T\left[\left(\sum_{i\in I_j}g_i\right) w_j +\frac{1}{2}\left(\sum_{i\in I_j}h_i+\lambda \right)w_j^2 \right] + \gamma \cdot T +C
\end{align}
$$



说明：

1. ​     $g_if_t(x_i)=g_iw_{q(x_i)}=(\sum_{i\in I_j}g_i)w_j$

   $f_t(x_i)$即第$t$个节点$x_i$样本的预测值，最后$(\sum_{i\in I_j}g_i)w_j$即：落在第$I_J$叶子节点中的样本的预测值加和
   
2. $w_{q(x_i)}$：叶子节点q的分数，预测值

3. $T$：叶子节点个数 



### 4.对目标函数继续简化

对于$J(f_t)=\sum_{j=1}^T\left[\left(\sum_{i\in I_j}g_i\right) w_j +\frac{1}{2}\left(\sum_{i\in I_j}h_i+\lambda \right)w_j^2 \right] + \gamma \cdot T +C$

定义 $G_j=\sum_{i\in I_j}g_i, \,H_J=\sum_{i\in I_J}h_j$

从而，$J(f_t)=\sum_{j=1}^T\left[G_jw_j+\frac{1}{2}(H_j+\lambda)w_j^2 \right]+\gamma\cdot T+c$

对$w$求偏导，得
$$
\frac{\partial J(f_t)}{\partial w_j}=G_j+(H_j+\lambda)w_j==0\implies w_j=-\frac{G_j}{H_j+\lambda}
$$
带回目标函数，得$J(f_t)=-\frac{1}{2}\sum_{j=1}^T\frac{G_j^2}{H_j+\lambda}+\gamma \cdot T$

![image-20200216145744529](pic\image-20200216145744529.png)

### 总结

XGBoost相对于传统的GBDT，XGBoost使用了二阶信息，可以更快的在训练集上收敛。

XGBoost的实现中使用了并行计算，因此训练速度更快，同时它的原生语言为c/c++，



## Adaboost

Adaboost可以看作是采用指数损失函数的提升方法，其每个基函数的学习算法为前向分步算法；

AdaBoost的训练误差是一直输速度下滑的；

AdaBoost算法不需要事先知道下届$\gamma$，具有自适应性（Adaptive），他能自适应弱分类器的训练误差率







## 总结

Bagging能够减少训练方差（Variance），对于不剪枝的决策树、神经网络等学习器有良好的集成效果；

Boosting能够减少偏差（Bias），能够基于泛化能力较弱的学习器构造强学习器。

除了GBDT中使用关于分类器的一阶导数进行学习之外，也可以借鉴（拟）牛顿法的思路，使用二阶导数学习弱分类器。



# 集成学习

- 容易过拟合的分类器：高方差低偏差的分类器 
- 容易欠拟合的分类器：低方差高偏差的分类器 
- 优化思路1：降低过拟合分类器的方差-
  - Bagging 类
  - 将一系列容易过拟合的分类器取平均 
  - 例子：随机森林
  - 例子：DropOut 
- 优化思路2：降低欠拟合分类器的偏差 
  - Boosting 类 
  - 将一系列容易欠拟合的分类器取加权平均 
  - 例子：广度神经网络 
  - 例子：Adaboost