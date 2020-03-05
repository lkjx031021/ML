import numpy as np
import pandas as pd

class ALS(object):

    def __init__(self, learn_rate=0.001,max_step=100,feature_len=2,random_state=0,penalty='',train_data_rate=0.75,C=1.0, max_tj_len= 10):
        '''
        penalty: 'l1' : l1正则化； 'l2'：l2正则化  目前只做了l2正则
        C : C值越大正则化效果越弱
        '''
        assert C > 0, 'C必须是大于0数字'
        self.learn_rate = learn_rate
        self.max_step = max_step
        self.feature_len = feature_len
        self.train_data_rate = train_data_rate
        self.penalty = penalty
        self.max_tj_len = max_tj_len
        self.lamda = 1 / np.log(C + 1)
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.RMSE_Train_Score = []
        self.RMSE_Test_Score = []

    def train(self,df_u_p):
        assert type(df_u_p).__name__ == 'DataFrame', '必须是DataFrame格式'
        self.user_ls = df_u_p.index.values
        self.product_ls = df_u_p.columns.values
        fill_1_mat = np.asmatrix(df_u_p.replace(0,1))
        u_p_mat = np.asmatrix(df_u_p.values)
        drop_0_mat = u_p_mat / fill_1_mat # 原始矩阵中只有大于0的部分才参与训练，此矩阵用来剔除用户商品评分为0的部分
        self.shape0, self.shape1 = u_p_mat.shape
        split_data_mat = np.random.random(u_p_mat.shape) # 用于区分训练集
        self.train_mat = np.where(split_data_mat < self.train_data_rate,1,0) # 训练集
        self.test_mat = np.where(self.train_mat == 1, 0, 1)                  # 测试集

        u_p_mat = np.multiply(u_p_mat,self.train_mat)
        u_p_test_mat = np.multiply(u_p_mat, self.test_mat)
        print(np.sum(u_p_mat))
        print(np.sum(drop_0_mat))
        print(drop_0_mat)
        # exit()
        # 初始参数随机设置
        self.user_feature = np.asmatrix(np.random.random([self.shape0, self.feature_len]))
        self.product_feature = np.asmatrix(np.random.random([self.feature_len, self.shape1]))
        if not self.penalty:
            self.lamda = 0
        for step in range(self.max_step):
            for feature_itr in range(self.feature_len):
                # 训练用户特征参数
                mat_cha = u_p_mat - np.dot(self.user_feature,self.product_feature)
                grad = -2 * np.sum(np.multiply(np.multiply(mat_cha,self.product_feature[feature_itr,:]),drop_0_mat), axis=1) \
                       + 2 *self.lamda * self.user_feature[:,feature_itr]
                self.user_feature[:,feature_itr] -= self.learn_rate * grad

                # 训练商品特征参数
                mat_cha = u_p_mat - np.dot(self.user_feature,self.product_feature)
                grad = -2 * np.sum(np.multiply(np.multiply(mat_cha, self.user_feature[:,feature_itr]),drop_0_mat), axis=0) \
                       + 2 *self.lamda * self.product_feature[feature_itr,:]
                self.product_feature[feature_itr,:] -= self.learn_rate * grad

            # 保存误差
            print('%d / %d' % (step, self.max_step))
            if step % 3 == 0:
                mat_hat = np.dot(self.user_feature, self.product_feature)
                self.RMSE_Train_Score.append(self.RMSE(np.multiply(u_p_mat, drop_0_mat), np.multiply(mat_hat,drop_0_mat),self.shape0,self.shape1))
                self.RMSE_Test_Score.append(self.RMSE(np.multiply(u_p_test_mat, drop_0_mat), np.multiply(mat_hat,drop_0_mat),self.shape0,self.shape1))

                print(self.user_feature)

        self.step = range(len(self.RMSE_Train_Score))
        self.user_feature = self.user_feature.astype(np.float16)
        self.product_feature = self.product_feature.astype(np.float16)
        print(self.user_feature)
        print(self.product_feature)
        self.result_mat = np.dot(self.user_feature, self.product_feature)
        self.result_mat = np.multiply(self.result_mat, 1 - drop_0_mat) # 将用户购买过的商品置为0
        df_result = pd.DataFrame(self.result_mat,index=self.user_ls,columns=self.product_ls)
        print(df_result.describe())
        print(df_result.info())
        self.df_result = df_result.T
        self.user_product_dt = self.df_result.to_dict()
        self.turn_dict()
        # for i in self.user_product_dt.keys():
        #     self.user_product_dt[i] = list(zip(*sorted(self.user_product_dt[i].items(),key=lambda x:x[1], reverse=True)))[0]
        del df_result

    def turn_dict(self):
        for i in self.user_product_dt.keys():
            pro_ls, prob = list(zip(*self.user_product_dt[i].items()))
            pro_ls, prob = np.array(pro_ls), np.array(prob)
            prob_rank = np.argsort(-prob)
            pro_ls = pro_ls[prob_rank][prob[prob_rank] > 0.15][:self.max_tj_len]
            # self.user_product_dt[i] = list(zip(*sorted(self.user_product_dt[i].items(),key=lambda x:x[1], reverse=True)))[0][:self.max_tj_len]
            self.user_product_dt[i] = pro_ls.tolist()[:15]

    def predict(self,user_id):
        return self.user_product_dt.get(user_id, [])

    def get_all_result(self):
        return self.user_product_dt

    def RMSE(self,mat1,mat2,m,n):
        return np.sqrt(np.square(mat1 - mat2) / m / n).sum()

    def save_model(self,file_path='../model/ALS_Model'):
        self.df_result.to_pickle(file_path)
        # np.savez('../model/ALS_Param.npz',als_user_mat=self.user_feature, als_pro_mat=self.product_feature,
        #          user_index=self.user_ls,pro_index=self.product_ls)

    def load_model(self,file_path='../model/ALS_Model'):
        self.df_result = pd.read_pickle(file_path)
        self.user_product_dt = self.df_result.to_dict()
        self.turn_dict()
        # for i in self.user_product_dt.keys():
        #     self.user_product_dt[i] = list(zip(*sorted(self.user_product_dt[i].items(),key=lambda x:x[1], reverse=True)))[0]
        del self.df_result

    def load_param(self, user_ls, file_path='../model/ALS_Param.npz'):
        data = np.load(file_path)
        self.user_feature = data['als_user_mat']
        self.product_feature = data['als_pro_mat']
        self.user_ls = data['user_index']
        self.product_ls = data['pro_index']
        mat = np.dot(self.user_feature[user_ls, :],self.product_feature)
        result = pd.DataFrame(mat,index=user_ls,columns=self.product_ls)
        result = result.T
        self.user_product_dt = result.to_dict()
        self.turn_dict()
        del self.user_feature, self.product_feature,



