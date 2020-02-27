## DeepFM模型

DeepFM包含两部分，神经网络部分与FM部分，分别负责缔结特征的提取和高阶特征的提取，这两部分共享同样的输入。DeepFM的预测结果为：
$$
\hat y = sigmoid(y_{FM}+y_{DNN})
$$
FM部分是因子分解机，其输出公式为：
$$
y_{FM}=<w,x>+\sum_{i=1}^{d}\sum_{j=i+1}^{d}<V_i,V_j>x_i\cdot x_J
$$
深度部分：

深度部分是全连接网络，由于数据的输入是非常稀疏的，因此需要引入嵌入层完成将输入的稀疏向量转化为低维稠密向量（所有特征embedding之后的向量维度均为K）。FM中的隐向量$V_i$即做为第$i$个特征embedding之后的向量

## FM部分 公式推导

$$
\begin{align}
&\,\,\,\,\,\,  \sum_{i=1}^{n}\sum_{j=i+1}^{n}<v_i,v_j>x_i x_J \\
&=\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}<v_i,v_j>x_i x_j -\frac{1}{2}\sum_{i=1}^{n}<v_i,v_i>x_ix_i\\
&=\frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{f=1}^kv_{if}v_{jf}x_i x_j -\frac{1}{2}\sum_{i=1}^{n}\sum_{f=1}^kv_{if}^2x_i^2\\
&=\frac{1}{2}\sum_{f=1}^k\left(\left(\sum_{i=1}^nv_{if}x_i\right)^2 -\sum_{i=1}^nv_i^2x_i^2 \right)

\end{align}
$$

## Deep 部分

```python
self.y_deep = tf.reshape(self.embeddings,shape=[-1,self.field_size * self.embedding_size])
```

将隐向量embedding矩阵[feature_size, embedding_size] reshape为[-1, feature_size * embedidng_size] 作为全连接的输入

最后将 线性部分 FM部分 和Deep部分拼接[linear, FM, Deep]