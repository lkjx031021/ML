"""
分词参考模型
"""

import tensorflow as tf 
import os 
import numpy as np 
class Data():
    def __init__(self):
        """
        初始化数据读取
        """
        base_dir = "./"
        #读取文件
        files = open("pku_training.utf8", "r", encoding="utf-8")
        datas = files.read().replace("\n", "  ")
        files_pt = open("pku_training_pt.utf8", "r", encoding="utf-8")
        datas_pt = files_pt.read().replace("\n", "  ")
        data_list = datas.split('  ')
        data_pt_ls = datas_pt.split('  ')
        print(data_pt_ls[:30])
        print(len(data_pt_ls))

        def label(txt):
            """
            将词数据转换为标签
            txt:文本词
            """
            len_txt = len(txt)
            if len_txt == 1:
                ret = 's'
            elif len_txt == 2:
                ret = 'be'
            elif len_txt > 2:
                mid = 'm'*(len_txt-2)
                ret = 'b'+mid+'e'
            else:
                ret = ''
            return ret 
        #设置文本标签
        data_labl = [label(itr) for itr in data_list]
        data_pt_labl = [label(itr) for itr in data_pt_ls]
        datas = ''.join(data_list)
        label = ''.join(data_labl)
        datas_pt = ''.join(data_pt_ls)
        label_pt = ''.join(data_pt_labl)
        # words_pt_set = set(datas_pt)
        words_set = set(datas)
        label2id = {'s':0,'b':1,'m':2,'e':3, 'o':4}
        self.id2label = {0:'s', 1:'b', 2:'m', 3:'e', 4:'o'}
        #保存字典词转换为id
        if os.path.exists(os.path.join(base_dir, "words2id5")) == False:
            self.word2id = dict(zip(words_set, range(1, len(words_set)+1)))
            with open(os.path.join(base_dir, "words2id5"), "w", encoding="utf-8") as f:
                f.write(str(self.word2id))
        else:
            with open(os.path.join(base_dir, "words2id5"), "r", encoding="utf-8") as f:
                self.word2id = eval(f.read())
        #保存字典id转换为词
        if os.path.exists(os.path.join(base_dir, "id2word5")) == False:
            self.id2word = dict()
            for itr in self.word2id:
                self.id2word[self.word2id[itr]] = itr
            with open(os.path.join(base_dir, "id2word5"), "w", encoding="utf-8") as f:
                f.write(str(self.id2word))
        else:
            with open(os.path.join(base_dir, "id2word5"), "r", encoding="utf-8") as f:
                self.id2word = eval(f.read())
        self.words_len = len(self.word2id) + 1
        self.data_ids = np.array([self.word2id.get(itr, 0) for itr in datas])
        self.data_pt_ids = np.array([self.word2id.get(itr, 0) for itr in datas_pt])
        self.label_ids = np.array([label2id.get(itr, 0) for itr in label])
        self.label_pt_ids = np.array([label2id.get(itr, 0) for itr in label_pt])
        self.seqlen = len(self.data_ids)
        np.savez("words_seg.npz", data=self.data_ids, label=self.label_ids)
    def next_batch(self, batch_size=32):
        """
        获取训练数据
        固定长度为50
        """
        length = 50
        x = np.zeros([batch_size, length])
        d = np.zeros([batch_size, length])
        for itr in range(batch_size):
            idx = np.random.randint(0, self.seqlen-length)
            x[itr, :] = self.data_ids[idx:idx+length]
            d[itr, :] = self.label_ids[idx:idx+length]
        return x, d
    def next_batch_pt(self, batch_size=32):
        """
        获取训练数据
        固定长度为50
        """
        length = 50
        self.length = len(self.data_pt_ids)
        x = np.zeros([batch_size, length])
        d = np.zeros([batch_size, length])
        for itr in range(batch_size):
            idx = np.random.randint(0, self.length - length)
            # print(self.data_pt_ids[idx:idx+length])
            x[itr, :] = self.data_pt_ids[idx:idx+length]
            d[itr, :] = self.label_pt_ids[idx:idx+length]
        return x, d
    def w2i(self, txt):
        """
        txt:一段文本
        词转换为ID
        """
        data = [self.word2id.get(itr, 0) for itr in txt] 
        data = np.array(data) 
        data = np.expand_dims(data, 0)
        return data
    def i2l(self, data):
        """
        data:标签id序列
        输出标签转换
        """
        return ''.join([self.id2label.get(itr, 's') for itr in data])


class Model():
    """
    分词模型中使用双向RNN模型进行处理
    """
    def __init__(self, is_training=True):
        """
        初始化类
        """
        self.is_training = is_training
        self.data_tools = Data() 
        self.words_len = self.data_tools.words_len
        self.build_model()
        self.init_sess(tf.train.latest_checkpoint("model/tfmodel-basic"))
    def build_model(self):
        """
        构建计算图
        """
        n_layers = 2
        hidden_size = 64
        embedding_size = 128
        self.graph = tf.Graph()

        rnn_cell = tf.nn.rnn_cell.BasicRNNCell
        # rnn_cell = tf.nn.rnn_cell.BasicLSTMCell
        # rnn_cell = tf.nn.rnn_cell.GRUCell
        with self.graph.as_default():

            self.inputs = tf.placeholder(name='inputs', shape=[None, None], dtype=tf.int32)
            self.target = tf.placeholder(name='target', shape=[None, None], dtype=tf.int32)
            self.seqlen = tf.placeholder(name='seqlen', shape=[None], dtype=tf.int32)
            self.mask = tf.placeholder(name='mask', shape=[None,None], dtype=tf.float32)
            self.weight = tf.placeholder(name='weight', dtype=tf.float32)

            emb = tf.get_variable('emb', [self.words_len, embedding_size])
            emb_inputs = tf.nn.embedding_lookup(emb, self.inputs)

            fw_cell_list = [rnn_cell(hidden_size) for _ in range(n_layers)]
            bw_cell_list = [rnn_cell(hidden_size) for _ in range(n_layers)]
            fw_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell_list)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell_list)

            (fw_output, bw_output), stats = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, emb_inputs,
                sequence_length=None,
                dtype=tf.float32
            )
            outputs = tf.concat([fw_output, bw_output], axis=2)

            # lstm_cell = [tf.nn.rnn_cell.BasicRNNCell(hidden_size) for _ in range(n_layers)]
            # lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cell)
            # outputs, stats = tf.nn.dynamic_rnn(lstm_cell, emb_inputs, dtype=tf.float32)

            print(outputs)
            self.logits = tf.layers.dense(outputs,5)
            print(self.logits)
            exit()
            self.loss = self.weight * tf.contrib.seq2seq.sequence_loss(
                self.logits,
                self.target,
                self.mask
            )
            self.loss = tf.reduce_mean(self.loss)
            self.step = tf.train.AdamOptimizer().minimize(self.loss)
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()


    def init_sess(self, restore=None):
        """
        初始化会话
        """
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        if restore != None:
            self.saver.restore(self.sess, restore)
    def train(self):
        """
        训练函数
        """
        import time
        for itr in range(200000):
            inx, iny = self.data_tools.next_batch(32)
            inx_, iny_ = self.data_tools.next_batch_pt(32)
            loss, _ = self.sess.run([self.loss, self.step],
                                    feed_dict={
                                        self.inputs: inx_,
                                        self.target: iny_,
                                        self.seqlen: np.ones(32) * 50,
                                        self.mask: np.ones([32, 50]),
                                        self.weight: np.array([10]) * 50
                                    }
                                    )
            loss, _ = self.sess.run([self.loss, self.step],
                feed_dict={
                    self.inputs:inx,
                    self.target:iny,
                    self.seqlen:np.ones(32) * 50, 
                    self.mask:np.ones([32, 50]),
                    self.weight: np.array([1])
                }
                                    )

            if itr%100==90:
                print(itr, loss)
                self.saver.save(self.sess, "model/segnet")
                # self.predict("实现祖国完全统一，是海内外一切爱国的中华儿女的共同心愿。")
                # self.predict("瓜子二手车直卖网，没有中间商赚差价。车主多卖钱，买家少花钱")
                self.predict("慕松庄园干红葡萄酒")
                self.predict("欧百乐私家特藏干红葡萄酒")
                self.predict("魔叶珍藏赤霞珠干红葡萄酒")
                self.predict("星云剑鱼座西拉干红")
                self.predict("拉菲瑞特干红")
                self.predict("拉菲传奇波尔多干红")

    #def predict(self, txt):
    def predict(self, txt):
        """
        预测函数
        txt:文本序列
        """
        inx = self.data_tools.w2i(txt) 
        out = self.sess.run(tf.argmax(self.logits, 2),
                            feed_dict={self.inputs:inx})
        seq = []
        s = ''
        last_label = ''
        for a, b in zip(txt, self.data_tools.i2l(out[0])):
            seq.append(a+b)
            if b == 's':
                if last_label:
                    s += ' ' +a + ' '
                else:
                    s += a + ' '
            elif b == 'b':
                s += ' ' + a
            elif b == 'm' and last_label == 'b':
                s += a
            elif b == 'm' and last_label not in ('b', 'm'):
                s += ' ' + a
            elif b == 'e' and last_label not in ('b', 'm'):
                s += ' ' + a
            else:
                s += a
            last_label = b

        print('|'.join(seq))
        print(s)
model = Model()
model.train() 
