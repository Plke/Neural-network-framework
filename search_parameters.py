from loss_function import *
import numpy as np
from optimizer import *
from network_layer import *
from help_function import *


def train_for_search(train_x, train_y, test_x, test_y, layers, loss, batch_size=100, eta=0.001, epoch=50,
                     optimizer=Adagrad, alpha=0.9):
    #每次搜索前，先对网络可学习参数重置
    layers = remake_model(layers)
    n_train = train_x.shape[0]  # 训练数据的采样数
    n_test = test_x.shape[0]  # 测试数据的采样数
    n_batch = n_train // batch_size

    for i in range(epoch):

        # -- 学习 --
        # 随机选取样本进行学习
        index_rand = np.arange(n_train)
        np.random.shuffle(index_rand)
        for j in range(n_batch):
            mb_index = index_rand[j * batch_size: (j + 1) * batch_size]
            x = train_x[mb_index, :]
            t = train_y[mb_index, :]

            a, z = forward_propagation(layers, x, True)

            grad_w, grad_b = backpropagation(layers, t, a, z, loss)
            uppdate_wb(layers, grad_w, eta, optimizer, grad_b, alpha)

    x, y, a = forward_sample(layers, train_x, train_y, n_train)
    count_train = np.sum(np.argmax(a, axis=1) == np.argmax(y, axis=1))

    x, y, a = forward_sample(layers, test_x, test_y, n_test)
    count_test = np.sum(np.argmax(a, axis=1) == np.argmax(y, axis=1))

    return count_test / n_test *0.5+ count_train / n_train*0.5


# -- 自动搜索参数 --
# 通过传入验证集，寻找较好的模型，可能因为数据量较小，训练次数较少等原因，存在较大的偶然性，误差
class search_para():
    def __init__(self, x, y, layers, lr=0.01, batch_size=10, epoch=50, loss=cross_entropy, optizmer=SGD, alpha=0.9):
        n_data = x.shape[0]  # 数据量
        index = np.arange(n_data)
        # 选取四分之三作为训练，四分之一作为测试
        index_train = index[index % 4 != 0]
        index_test = index[index % 4 == 0]
        self.train_x = x[index_train, :]  # 训练 输入数据
        self.train_y = y[index_train, :]  # 训练 正确答案
        self.test_x = x[index_test, :]  # 测试 输入数据
        self.test_y = y[index_test, :]  # 测试 正确答案

        self.layers = layers
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.loss = loss
        self.optizmer = optizmer
        self.max_acc = 0
        self.alpha = alpha

    # 寻找学习率，批大小，学习次数
    def search_lr(self, lr_list, batch_list, epoch_list):
        for lr in lr_list:
            for batch_size in batch_list:
                for epoch in epoch_list:

                    acc = train_for_search(self.train_x, self.train_y, self.test_x, self.test_y, self.layers, self.loss,
                                           batch_size, lr, epoch, self.optizmer, self.alpha)
                    if acc > self.max_acc:
                        self.lr = lr
                        self.batch_size = batch_size
                        self.epoch = epoch
                        self.max_acc = acc
        return self.lr, self.batch_size, self.epoch

    # 寻找输出层的激活函数和损失函数的搭配
    def search_outlayer_loss_activation(self, loss_list, activation_list):
        for activation in activation_list:
            # 替换最后一层
            temp_layers = self.layers[:-1]
            temp_layers.append(OutputLayer(self.layers[-1].i, self.layers[-1].o, activation))
            for loss in loss_list:
                acc = train_for_search(self.train_x, self.train_y, self.test_x, self.test_y, temp_layers, loss,
                                       self.batch_size, self.lr, self.epoch, self.optizmer, self.alpha)
                if acc > self.max_acc:
                    self.loss = loss
                    self.layers = temp_layers
                    self.max_acc = acc
        return self.loss, self.layers

    # 寻找优化器
    def search_optizmer(self, optizmer_list):
        for optizmer in optizmer_list:
            acc = train_for_search(self.train_x, self.train_y, self.test_x, self.test_y, self.layers, self.loss,
                                   self.batch_size, self.lr, self.epoch, optizmer, self.alpha)
            if acc > self.max_acc:
                self.optizmer = optizmer
                self.max_acc = acc
        return self.loss, self.optizmer
    #寻找网络结构
    def search_layers(self, layers_list):
        for layers in layers_list:
            acc = train_for_search(self.train_x, self.train_y, self.test_x, self.test_y, layers, self.loss,
                                   self.batch_size, self.lr, self.epoch, self.optizmer, self.alpha)
            if acc > self.max_acc:
                self.layers = layers
                self.max_acc = acc
        return self.loss, self.optizmer

    def get_all_param(self):
        return self.layers, self.lr, self.batch_size, self.epoch, self.loss, self.optizmer
