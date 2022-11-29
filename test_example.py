from help_function import *
import matplotlib.pyplot as plt
from loss_function import *
from  activation_function import *
from network_layer import *
from  Performance import *
from optimizer import *
from train import *

layers = np.load('layer.npy', allow_pickle=True)

from sklearn import datasets

train_x=np.load('data/train_x.npy').T
train_y=np.load('data/train_y.npy').T
test_x=np.load('data/test_x.npy').T
test_y=np.load('data/test_y.npy').T

n_train = train_x.shape[0]  # 训练数据的采样数
n_test = test_x.shape[0]  # 测试数据的采样数

train_x=train_x.reshape(n_train,1,28,28)
test_x=test_x.reshape(n_test,1,28,28)

a, _ = forward_propagation(layers, test_x, False)

acc = np.sum(np.argmax(a, axis=1) == np.argmax(test_y, axis=1))/n_test

print(acc)
