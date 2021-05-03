# 代码demo2
# -*- coding: utf-8 -*-
# @Time    : 2020/5/11 11:18
import collections
import datetime
import time
import numpy as np
import torch
import keras
import pandas as pd
from torch import nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from keras import backend as K
# from tensorboardX import SummaryWriter
DAYS_FOR_TRAIN = 10
EPOCHS = 1000
import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from prettytable import PrettyTable
class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_auc = []
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_accuracy = []


    def on_epoch_end(self, epoch, logs={}):

        val_predict = (numpy.asarray(self.model.predict(
            self.validation_data[0]))).round()[:,::14,1]
        val_targ = self.validation_data[1][:,::14,1]
        _val_auc = roc_auc_score(val_targ,val_predict)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_accuracy = accuracy_score(val_targ,val_predict)
        self.val_auc.append(_val_auc)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_accuracy.append(_val_accuracy)
        return

def binary_focal_loss(gamma=2, alpha=0.1):
    """
    Binary form of focal loss.
    适用于二分类问题的focal loss

    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

        p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)

    return binary_focal_loss_fixed


if __name__ == '__main__':

    df = pd.read_csv(r"C:\\Users\\office\\Desktop\\data\\factorData\\allAts.csv")
    factor_to_use = pd.read_csv(r"C:\\Users\\office\\Desktop\\data\\factor_importance_new0201.csv", usecols=[0],
                                encoding="gbk")
    level = pd.read_csv(r"C:\\Users\\office\\Desktop\\data\\levelRateNO300.csv")
    scaler = StandardScaler()
    numberList = [10,20,30,40,50,60,70,80,90,100]
    dropOutList = [0.1,0.2,0.3,0.4,0.5]
    batchSizeList = [8,16,32,64]

    for factorNumber in numberList:
        factorList = factor_to_use.values[0:factorNumber,:]
        # data = pd.DataFrame(columns=['level'], data=df["level"].values.reshape(-1, 1))
        data = pd.DataFrame(columns=['level'], data=level["level_0.09"].values.reshape(-1, 1))
        for factor in factorList:
            temp = scaler.fit_transform(df[factor[0]].values.reshape(-1, 1))
            data[factor[0]] = temp
        data = data.fillna(method='ffill')  # ffill指用向前法填充缺失值
        data = np.array(data)

        data_x = data[:, 1:]
        data_y = data[:, 0]

        sqe_len = 14#一组数据为14天
        stock_lenth = len(data)
        tr_val_slip = int(0.8 * stock_lenth / 14)  # 测试集中数据组数

        print("stock_lenth = ", stock_lenth)
        num1 = int((stock_lenth / 14))#总共数据的组数
        x = np.zeros((num1, sqe_len, factorNumber))#输入为num1组，每组sqe_len个数据，每个数据包含number个因子
        y = np.zeros((num1, sqe_len, 2))
        # print(data[0:14][:,0:6])
        # print(type(data[0:14]))
        i = 0
        while i < stock_lenth:
            x[int(i / 14)] = data_x[i: i + sqe_len]
            y_tmp = data_y[i: i + sqe_len].reshape(-1, 1)  ##转成one-hot encoding
            y[int(i / 14)] = keras.utils.to_categorical(y_tmp, 2)  # 用kearas
            i += 14
        print(x.size," ",y.size)
        train_x = x[0:tr_val_slip]
        train_y = y[0:tr_val_slip]
        valid_x = x[tr_val_slip:]
        valid_y = y[tr_val_slip:]
        for dropOut in dropOutList:
            for batchSize in batchSizeList:
                cur = datetime.datetime.now()
                print("============factorNumber=",factorNumber,"=====dropOut =", dropOut, "====batchSize = ",batchSize,"训练开始=====================")

                model = Sequential()
                model.add(LSTM(units=512, input_shape=(train_x.shape[1], train_x.shape[2]),
                               return_sequences=True))  # unit是神经元，shape【1】是序列长度也就是14，第二个维度就是每天的因子数目，第三个参数表示返回的是序列还是值
                model.add(LSTM(units=512, activation='relu', return_sequences=True))  # activation是激活函数
                model.add(LSTM(units=512, activation='relu', return_sequences=True))#用三层lstm是因为感觉可以这么用，也可以多用几层，可以用relu也可以用tanh
                model.add(Dropout(dropOut))  # 防止过拟合，随机的挑选出一半神经元，其他的就舍弃掉
                model.add(Dense(128, activation='relu'))
                # model.add(Dropout(dropOut))
                model.add(Dense(64, activation='relu'))
                # model.add(Dropout(dropOut))
                model.add(Dense(2, activation='sigmoid'))  # 期望得到2维的东西，所以这个不能改
                # model.add(Dropout(dropOut))

                model.compile(loss='binary_crossentropy', optimizer='adam',
                              metrics=['accuracy'])  # 设置损失函数，metrics是评价指标，optimizer是优化器一半不改，loss是损失函数
                # model.add_loss("l2")
                # fit network

                # history = model.fit(train_x, train_y, epochs=60, batch_size=batchSize, validation_data=(vaild_x, valid_y), verbose=2,
                #                     shuffle=True)  # epochs是训练多少次，batch_size是一次训练多少数据，loss都是根据一个batch_size数量的数据得出来的

                metrics = Metrics()
                history = model.fit(train_x, train_y, epochs=10, batch_size=batchSize,
                verbose = 2, validation_data = (valid_x, valid_y),
                callbacks = [metrics])
                table = PrettyTable(
                    ["epoch", "accuracy_score", "precision_score", "recall_score", "F1_score", "AUC_score"])
                for (index) in (range(len(metrics.val_f1s))):
                    table.add_row(
                        [index, round(metrics.val_accuracy[index], 4), round(metrics.val_precisions[index], 4),
                         round(metrics.val_recalls[index], 4), round(metrics.val_f1s[index], 4),
                         round(metrics.val_auc[index], 4)])
                    # print(" epoch: ",index, " precision_score: ",metrics.val_precisions[index]," recall_score: ",metrics.val_recalls[index]," f1_score: ",metrics.val_f1s[index])

                print(table)
                # 绘制损失图
                # print(history.history.keys())
                # plt.figure()
                # plt.plot(history.history['accuracy'], label='train')
                #
                # plt.plot(history.history['val_accuracy'], label='test')
                #
                # plt.title('LSTM_600000.SH---dropOut = ' + str(dropOut) + "batchSize = " + str(batchSize), fontsize='12')
                #
                # plt.ylabel('accuracy', fontsize='10')
                #
                # plt.xlabel('epoch', fontsize='10')
                #
                # plt.legend()
                # pngName = "factorNumber="+str(factorNumber)
                # pngName = "C:\\Users\\Ats\\Desktop\\" + pngName
                # plt.savefig(pngName)



                ### 输出预测结果
                print("===============factorNumber=",factorNumber,"==================")
                # yVal = model.predict(vaild_x, verbose=0)
                # print(collections.Counter(np.argmax(yVal[:, -1, :], axis=1)))
                # m, s = divmod((datetime.datetime.now() - cur).seconds, 60)
                # h, m = divmod(m, 60)
                # print("总耗时： %02d:%02d:%02d" % (h, m, s))
                # print("============factorNumber=",factorNumber,"dropOut =", dropOut, "====batchSize = ",batchSize,"==================预测结果")

                # model.save("dropOut = " + str(dropOut) + "batchSize = " + str(batchSize) + ".h5")


