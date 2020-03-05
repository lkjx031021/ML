## 1、FFM

离散特征需要转换为one-hot编码，转换为one-hot编码后原来某个特征下的属性值全部变为新的特征，但是这些特征都属于同一个field。

每个特征针对每个field都有一个长度为$K$的需要学习的隐向量，总共可训练参数为n * f * k个。 n为特征数量，f为字段数量

| 字段-field | 价格 | 品种   | 品种 | 国家 | 国家 |
| ---------- | ---- | ------ | ---- | ---- | ---- |
| 特征       | 价格 | 赤霞珠 | 西拉 | 法国 | 澳洲 |
| 商品1      | 23.8 | 1      | 0    | 0    | 1    |
| 商品2      | 45.9 | 1      | 1    | 1    | 0    |


$$
\hat y = w_0 + \sum_{i=1}^{n}w_ix_i + \sum_{i=1}^{n-1}\sum_{j=i+1}^{n}<v_{i,f_j},v_{j,f_i}>x_ix_j
$$

- $v_{i,f_j}$表示第$i$个特征对应第$f_j$个字段的向量（该向量长度为$K$）

$Logistic函数$：

$g(x) = \frac{1}{1+e^{-x}}$

$g^{'}(x) = g(x)(1-g(x))$



$P(y=1|x;\theta)=g(x)$

$P(y=0|x;\theta)=1-g(x)$

$P(y|x;\theta)=(g_{\theta}(x))^y(1-g_{\theta}(x))^{1-y}$



似然函数：

$L(\theta)=\prod_{i=1}^{n}(g_{\theta}(x))^y(1-g_{\theta}(x))^{1-y}$

对数似然：
$$
l(\theta)=-logL(\theta)=-\sum_{i=1}^{n}(y^ilogg_{\theta}(x_i) + (1-y^i)log(1-g_{\theta}(x_i)))
$$

$$
\begin{align}
\frac{\partial l}{\partial \theta} &= -\sum_{i=1}^{n}(y^i\frac{1}{g_\theta(x_i)} \frac{\partial {g_\theta(x_i)}}{\partial \theta} - (1-y^i)\frac{1}{1-g_\theta(x_i)}\frac{\partial g_\theta(x_i)}{\partial \theta}) \\
& = -\sum_{i=1}^{n}(y^i\frac{1}{g_\theta(x_i)} + y^i\frac{1}{1-g_\theta(x_i)} -  \frac{1}{1-g_\theta(x_i)})\frac{\partial g_\theta(x_i)}{\partial x_i}\frac{\partial x_i}{\partial \theta} \\ 
&= - \sum_{i=1}^{n}(y^i\frac{1}{g_\theta(x_i)(1-g_\theta(x_i))} - \frac{1}{1-g_\theta(x_i)}) g_\theta(x_i)(1-g_\theta(x_i))\frac{\partial x_i}{\partial \theta} \\
&= \sum_{i=1}^{n}(g_\theta(x_i) - y^i) \frac{\partial x_i}{\partial \theta}
\end{align}
$$

$$
\frac{\partial x_i}{\partial \theta} = \begin{cases} 1,  & \theta = w_0 \\ x_i, &\theta=w_i  \\ v_{i,f_j}x_ix_j,&\theta=v_{j,f_i}  	\end{cases}
$$

