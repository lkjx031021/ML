##  LFM（隐语义模型）

找到用户的偏好特征，将该类偏好对应特征对应的书籍推荐给用户

问题：

	1. 不好把控K的设置 只能尝试
 	2. 能计算出物品在各个类别中的权重，，这是根据用户的行为数据统计的，不会将其只归到1类。



缺点：

​	1. 不能做到实时推荐



将用户产品矩阵分解为两个较小的矩阵相乘，

$R=\begin{array}{c|lll}{\downarrow}&{I_{1}}&{I_{2}}&{I_{3}}&{I_{4}} \\\hline{U_1}&{0}&{1}&{2}&{0}\\{U_2}&{1}&{0}&{2.5}&{0}\\{U_2}&{0.5}&{0.5}&{0.5}&{0.5}\\{U_2}&{0}&{0}&{0}&{1}\\\end{array} =  \left[\begin{matrix} \theta_{11} & \theta_{12} \\ \theta_{21} & \theta_{22} \\ \theta_{31} & \theta_{32} \\ \theta_{41} &  \theta_{42} \end{matrix} \right]  \left[\begin{matrix} w_{11} & w_{12} & w_{13} & w_{14} \\ w_{21} & w_{22} & w_{23} & w_{24}  \end{matrix} \right]$



- $\theta矩阵$： 用户参数
- $w矩阵$：商品参数
- $R矩阵$：用户商品矩阵
- $R矩阵中的元素值$：用户对商品的评分，根据历史订单进行计算，$r_{vi} = \frac{\sum_{i\in\{op\}}M(i)W(o)}{M}T(i) $

在一段时间内

- $op$ : 用户v关于物品i的所有订单记录
- $M(i)$ : op中每一单下单金额
- $T(i) $: 用户v购买物品i的次数
- $W(o)$ : 订单类型对应的权重
- $M$ ： 用户v总下单金额



```当求得的参数矩阵能够很好的描述已有的用户商品矩阵时，就可以认为其能够较好地预测未知数据。```



#### 求解

$$
L = \sum_{u,i \in R}(r_{ui} - \theta_uW_i)^2 + \lambda_\theta\sum_u{||\theta_u||^2} + \lambda_W\sum_W{||W_i||^2} \\
\frac{\partial L}{\partial \theta_u} = -2\sum_i(r_{ui} - \theta_uW_i)*W_i  + 2\lambda_\theta\theta_u \\
\frac{\partial L}{\partial W_i} = -2\sum_i(r_{ui} - \theta_uW_i)*\theta_u  + 2\lambda_WW_i
$$

参数$\theta$ 和$W$ 初始值随机设定，固定其中一个参数的值求另一个参数，交替求解

####判断收敛：

$$
RMSE = \sqrt{\frac{\sum(R - \theta W^T)^2}{N}}
$$

- $N$: $R矩阵元素个数$