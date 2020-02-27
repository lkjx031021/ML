import numpy as np
import tensorflow as tf

a = np.arange(8).reshape(2, 4)
b = np.arange(8, 16).reshape(2, 4)
c = np.arange(12, 20).reshape(2, 4)
# a = np.array([0,1,2,3])
# b = np.array([8,9,10,11])
# c = np.array([12,13,14,15])

print ("a :")
print (a)
print ("b :")
print (b)
print ("c :")
print (c)
params = tf.constant(np.array([a,b,c]), dtype=tf.float64)

a = tf.Variable(a, dtype=tf.float32)
b = tf.Variable(b, dtype=tf.float32)
c = tf.Variable(c, dtype=tf.float32)

idx = tf.SparseTensor(indices=[[0,0], [0,2], [1,0], [1, 1]], values=[1,2,2,0], dense_shape=(2,3))
result = tf.nn.embedding_lookup_sparse(params, idx, None, partition_strategy='div', combiner="sum")

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    res = sess.run(result)
    print(sess.run(tf.sparse_tensor_to_dense(idx)))
    print ("\n# result here")
    print(res)

exit()
x = tf.sparse_placeholder(tf.float32)

params = tf.constant([[0.1, 0.4, 0.5, 7.0, 6.4, 1.2, 0.5, 0.3, 3.3, 2.0],
                      [0.3, 0.4, 0.9, 0.8, 0.5, 0.3, 0.7, 0.5, 0.8, 3.2],
                      [0.4, 0.9, 1.1, 4.3, 3.4, 0.2, 0.3, 0.2, 0.5, 0.1]])

ids = tf.SparseTensor(indices=[[0, 1],
                               [0, 3],
                               [1, 2],
                               [1, 3]],
                      values=[2, 1, 1, 1],
                      dense_shape=[2, 4])

with tf.Session() as sess:
    lookup = sess.run(
        tf.nn.embedding_lookup_sparse(params, ids, None,
                                      partition_strategy="div", combiner="sum"))

print(lookup)