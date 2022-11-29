from help_function import *
import matplotlib.pyplot as plt
from loss_function import *
from  activation_function import *
from network_layer import *
from  Performance import *
from optimizer import *
from train import *
from search_parameters import *

ml_1 = MiddleLayer(28 * 28, 256, relu)
dr_1 = Dropout(0.5)
ml_2 = MiddleLayer(256, 128, relu)
dr_2 = Dropout(0.5)
ol_1 = OutputLayer(128, 10, softmax)
layers = [ml_1, dr_1, ml_2, dr_2, ol_1]

train_x = np.load('train_x.npy').T / 255
train_y = np.load('train_y.npy').T
test_x = np.load('test_x.npy').T / 255
test_y = np.load('test_y.npy').T

n_train = train_x.shape[0]  # 训练数据的采样数
n_test = test_x.shape[0]  # 测试数据的采样数

train_x = train_x.reshape(n_train, 1, 28, 28)  # 数据转化为【n,channel,width,height】
test_x = test_x.reshape(n_test, 1, 28, 28)

best_para = search_para(train_x[0:int(n_train / 5)], train_y[0:int(n_train / 5)], layers, lr=0.01, batch_size=50,   epoch = 5, loss=cross_entropy, optizmer=SGD,
                        alpha=0.9)
loss, layers = best_para.search_outlayer_loss_activation([cross_entropy, mse], [sigmoid, softmax])
print(loss, layers[-1].activation)
