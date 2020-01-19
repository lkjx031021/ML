## EM算法

##### 第一种推导方式（$Jensen$不等式）：

假设观测数据$x=(x_1, x_2, x_3,\cdots, x_n)$，数据中含有隐藏变量$z=(z_1, z_2,\cdots)$其对数似然函数为：
$$
\begin{align}
l(\theta)=\sum_{i=1}^n\log \sum_jp(x,z|\theta) &=\sum_{i=1}^n\log \sum_{j}p(x_i,z_j|\theta) \tag{2} \\ 
&=\sum_{i=1}^m\log \sum_{j}Q(z_j|x_i)\frac{p(x_i,z_j|\theta)}{Q(z_j|x_i)} \tag{3} \\
&\ge \sum_{i=1}^m\sum_{j}Q(z_j|x_i)\log \frac{p(x_i,z_j|\theta)}{Q(z_j|x_i)} \tag{4}
\end{align}
$$

其中$Q$函数是$z$的某一个分布，$Q\ge 0$

$Jensen$不等式，因为$\log$函数为凹函数，所以：
$$
E(f(x)) \le f(E(x))
$$
公式$(3)$到$(4)$由$Jensen$不等式得到

当且仅当$f(x)$为常数时等号成立：
$$
\begin{align}
\text{条件：} 
\frac{p(x_i,z_j|\theta)}{Q(z_j)}=c \,\,\,,\, 
\sum_jQ(z_j)=1 \\
\rightarrow \sum_jp(x_i,z_j|\theta) = p(x_i|\theta) = c       \\
\rightarrow Q(z_j)=\frac{p(x_i,z_j|\theta)}{p(x_i|\theta)} \\
=p(z_j|x_i,\theta)
\end {align}
$$
EM算法步骤：

E步：

根据$\theta$和数据估计变量$z$的分布：
$$
Q(z)=p(z|x,\theta)  \tag{5}
$$
M步，使$\theta$最大：
$$
\theta := \arg \max_{\theta}\sum_i\sum_jQ(z_j)\log(\frac{p(x_i,z_i|\theta)}{Q(z_j)})
$$
M步后将更新的$\theta$带入到公式$(5)$中重复执行E步、M步



##### 第二种推导方式


$$
L_i=\log p(x_i)
$$

$$
LL_i=\log p(x_i)-\sum_j q(z_j|x_i)\log \frac{q(z_j|x_i)}{p(z_j|x_i)}\\LL_i=\sum_j \left[q(z_j)\log p(x_i)-q(z_j)\log \frac{q(z_j)}{p(z_j)}\right]\\LL_i=\sum_j \left[q(z_j)\log \frac{p(x_i)p(z_j)}{q(z_i)}\right]\\LL_i=\sum_j\left[q(z_j|x_i)\log\frac{p(x_i,z_j)}{q(z_j)}\right]
$$

$$
例子
硬币，假设有三个硬币ABC，先抛A，如果是正面则选B，否则选C，抛掷选择的硬币，出现正面记为1,否则记为0，\\
假设三个硬币正面概率为a,b,c。独立进行n次试验记录样本x。仅记录最终的01。\\

此时隐变量z为选择硬币B。\\

根据上面所说，\\
E步-隐藏变量分布\\
EM算法第一步E步，为观测隐藏变量分布。在假设的abc条件以及观测变量x_i的条件下计算抛硬币为B的概率：\\
$$


$$
\mu_i =p(B|x_i,\theta)=\frac{p(x_i|B, \theta)}{p(x_i|B, \theta)+p(x_i|\urcorner B, \theta)}=\frac{a b^{x_i}(1-b)^{1-
x_i}}{a b^{x_i}(1-b)^{1-
x_i}+(1-a)c^{x_i}(1-c)^{1-
x_i}}
$$

$$
M步-参数最大化
$$

$$
\begin{matrix}a = \frac{1}{n}\sum_i \mu_i\\b = \frac{\sum_i \mu_i y_i}{\sum_i{\mu_i}}\\c = \frac{\sum_i (1-\mu_i) y_i}{\sum_i{(1-\mu_i)}}\\\end{matrix}
$$

