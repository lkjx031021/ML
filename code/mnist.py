import tensorflow as tf
from keras.datasets import mnist
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

seed = 523
tf.set_random_seed(seed)
np.random.seed(seed)

# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# batch_size, h, w = x_train.shape
# x_train = x_train.reshape(batch_size, h, w, 1)
# x_test = x_test.reshape(x_test.shape[0], h, w, 1)

mnist = input_data.read_data_sets("", one_hot=True)# 读取图片数据集
x_train, y_train, x_test, y_test = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
x_train = x_train.reshape([-1, 28, 28, 1])
x_test = x_test.reshape([-1, 28, 28, 1])
print(y_train[:10])

def center_loss(emb, labels, num_class, name=''):
    with tf.get_variable_scope(name):
        batch_size, num_features = emb.get_shape().as_list()
        centerids = tf.get_variable(shape=[num_class, num_features], name='c', initializer=tf.random_normal_initializer(mean=0,stddev=0.5))
        label_index = tf.argmax(labels)
        center_emb = tf.gather(centerids, label_index)
        closs = tf.nn.l2_loss(emb - center_emb) / float(batch_size)
    return closs

class Mnist(object):

    def __init__(self,is_training, keep_prob, batch_size):
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.batch_size = batch_size
        self.build_network()
        pass

    def init_weights(self):
        pass

    def build_network(self):
        self.inputs = tf.placeholder(tf.float32,[None, 28, 28, 1])
        self.labels = tf.placeholder(tf.float32,[None, 10])
        # images = tf.reshape(self.inputs, [-1, 28, 28 ,1])

        with tf.variable_scope('conv1'):
            W_conv1 = self.weights_variable('w_conv1',[3, 3, 1, 32])
            # 对于每一个卷积核都有一个对应的偏置量。
            b_conv1 = self.bias_variable('b_conv1', [32])
            # 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
            conv1_out = tf.nn.relu(self.conv2D(self.inputs, W_conv1) + b_conv1)
            print(conv1_out)
            conv1_out = self.max_pool_2x2(conv1_out)
            print(conv1_out)
            conv1_out = tf.nn.dropout(conv1_out, self.keep_prob)
            print('--------------------')

        with tf.variable_scope('conv2'):
            W_conv2 = self.weights_variable('w_conv2',[3, 3, 32, 64],)
            # 对于每一个卷积核都有一个对应的偏置量。
            b_conv2 = self.bias_variable('b_conv2', [64])
            # 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
            conv2_out = tf.nn.relu(self.conv2D(conv1_out, W_conv2, strides=[1,1,1,1]) + b_conv2)
            print(conv2_out)
            conv2_out = self.max_pool_2x2(conv2_out)
            conv2_out = tf.nn.dropout(conv2_out, self.keep_prob)

        with tf.variable_scope('conv3'):
            W_conv3 = self.weights_variable('w_conv2', [3, 3, 64, 128])
            # 对于每一个卷积核都有一个对应的偏置量。
            b_conv3 = self.bias_variable('b_conv2', [128])
            # 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
            conv3_out = tf.nn.relu(self.conv2D(conv2_out, W_conv3) + b_conv3)
            conv3_out = self.max_pool_2x2(conv3_out)
            print(conv3_out)
            conv3_out = tf.reshape(conv3_out,[-1, 128*4*4])
            conv3_out = tf.nn.dropout(conv3_out, self.keep_prob)

        # conv_out = tf.layers.batch_normalization(conv_out,training=self.is_training)
        # # conv_out = tf.layers.flatten(conv_out)
        # print(conv_out.shape)

        with tf.variable_scope('full_connect'):
            w_fc1 = self.weights_variable('w_fc1', [2048, 625])
            b_fc1 = self.bias_variable('b_fc1', [625])
            # conv_out = tf.reshape(conv_out, [-1, 72])
            h_fc1 = tf.nn.relu(tf.matmul(conv3_out, w_fc1) + b_fc1)
            h_fc1 = tf.nn.dropout(h_fc1,keep_prob=self.keep_prob)

        # with tf.variable_scope('full_connect2'):
        #     w_fc2 = self.weights_variable('w_fc2', [625, 2])
        #     h_fc2 = tf.matmul(h_fc1,w_fc2)
        #
        # self.logits = h_fc2

        with tf.variable_scope('full_connect3'):
            w_fc3 = self.weights_variable('w_fc3', [625, 10])
            y_out = tf.matmul(h_fc1,w_fc3)
            y_out = tf.nn.softmax(y_out)

        print(self.labels.get_shape().as_list())
        print(y_out.get_shape().as_list())
        self.a1 = self.labels.get_shape().as_list()
        self.a2 = y_out.get_shape().as_list()

        self.cross_entropy = - self.labels * tf.log(y_out + 1e-10)
        self.cross_entropy = tf.reduce_mean(tf.reduce_sum(self.cross_entropy, axis=1))
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_out, labels=self.labels))
        self.train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(self.cross_entropy)
        # self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy)
        # self.cross_entropy = - self.labels * tf.log(y_out + 1e-10)
        # self.cross_entropy = tf.reduce_mean(tf.reduce_sum(self.cross_entropy, axis=1))
        # self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy)
        # self.train_step = tf.train.AdadeltaOptimizer(0.01).minimize(self.cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.labels,1), tf.argmax(y_out,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), axis=0)


    def fit(self):
        sess = tf.InteractiveSession()# 创建session
        sess.run(tf.global_variables_initializer())
        for itr in range(200):
            train_batch = zip(range(0, len(x_train), self.batch_size),
                              range(self.batch_size,len(x_train), self.batch_size))
            # start, end = 0, 100
            for start, end in train_batch:
                # print(start, end)
                # print(x_train[:128].shape)
                # print(y_train[:128].shape)

                _, loss, acc = sess.run([self.train_step, self.cross_entropy, self.accuracy],feed_dict={self.inputs:x_train[start:end],self.labels:y_train[start:end]})
                # _, loss = sess.run([self.train_step, self.cross_entropy],feed_dict={self.inputs:x_train[start:end],self.labels:y_train[start:end]})

            test_indices = np.arange(len(x_test))
            np.random.shuffle(test_indices)
            test_indices = test_indices[:1000]
            # batch = mnist.train.next_batch(32)
            # print(batch[0][1])
            # plt.matshow(batch[0][1].reshape([28,28]))
            # plt.show()
            # a = mnist.test.next_batch(1000)
            [test_acc] = sess.run([self.accuracy], feed_dict={self.inputs:x_test[test_indices], self.labels: y_test[test_indices]})
            # test_acc = sess.run(self.accuracy, feed_dict={self.inputs:a[0], self.labels: a[1]})
            print('step:{}, loss:{:.4f}, accuracy:{:.4f}, test_acc:{:.4f}'.format(itr,loss, acc, test_acc))
            # plt.scatter(logits[:, 0], logits[:, 1], c=np.argmax(y_test[test_indices], axis=1).flatten())
            # plt.show()
            # labs = np.argmax(y_test[test_indices],axis=1)
            # print(labs)
            # print(logits)
            # c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
            #      '#ff00ff', '#990000', '#999900', '#009900', '#009999']
            # for i in range(10):
            #     plt.scatter(logits[labs == i, 0].flatten(), logits[labs == i, 1].flatten(), c=c[i])
            # # plt.scatter(logits[:, 0], logits[:, 1], c=np.argmax(y_test[test_indices], axis=1).flatten())
            # plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            # plt.grid()
            # # plt.xlim(-2, 2)
            # # plt.ylim(-2, 2)
            # plt.savefig('result/'+str(itr) + '.png')
            # # plt.show()
            # plt.close()

        pass


    def weights_variable(self, name, shape):
        return tf.get_variable(name, shape, initializer=tf.random_normal_initializer(mean=0,stddev=0.01),dtype=tf.float32)

    def bias_variable(self, name, shape):
        return tf.get_variable(name, shape=shape,initializer=tf.zeros_initializer(), dtype=tf.float32)

    def conv2D(self, images, kernel, strides=[1, 1, 1, 1], padding="SAME"):
        return tf.nn.conv2d(images, filter=kernel, strides=strides, padding=padding)

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


if __name__ == '__main__':
    model = Mnist(is_training=1, keep_prob=0.5, batch_size=128)
    model.fit()

    print(111)