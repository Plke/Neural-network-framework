from help_function import *
import matplotlib.pyplot as plt
from loss_function import *
from  activation_function import *
from network_layer import *
from  Performance import *
from optimizer import *
from train import *

train_x = np.load('data/train_x.npy').T / 255
train_y = np.load('data/train_y.npy').T
test_x = np.load('data/test_x.npy').T / 255
test_y = np.load('data/test_y.npy').T

n_train = train_x.shape[0]  # 训练数据的采样数
n_test = test_x.shape[0]  # 测试数据的采样数

train_x = train_x.reshape(n_train, 1, 28, 28)  # 数据转化为【n,channel,width,height】
test_x = test_x.reshape(n_test, 1, 28, 28)
# plt.imshow(train_x[0,0],cmap='gray')


# -- 相关参数初始化 --
eta = 0.001  # 学习系数
epoch = 5
batch_size = 200
interval = 1  # 每次进行显示的间隔
n_sample = 200  # 每次进行误差计算的采样数

# 网络初始化

img_h = 28  # 输入图像的高度
img_w = 28  # 输入图像的宽度
img_ch = 1  # 输入图像的通道数

# 【图片通道，图片高度,图片宽度，滤波器数量，滤波器宽度，滤波器高度，stride, padding，激活函数】
cl_1 = ConvLayer(img_ch, img_h, img_w, 10, 3, 3, 1, 1, relu)
cl_2 = ConvLayer(cl_1.y_ch, cl_1.y_h, cl_1.y_w, 10, 3, 3, 1, 1, relu)
pl_2 = PoolingLayer(cl_2.y_ch, cl_2.y_h, cl_2.y_w, 2, 0)
n_fc_in = pl_2.y_ch * pl_2.y_h * pl_2.y_w
ml_1 = MiddleLayer(n_fc_in, 200, relu)
dr_1 = Dropout(0.5)
ml_2 = MiddleLayer(200, 200, relu)
dr_2 = Dropout(0.5)
ol_1 = OutputLayer(200, 10, softmax)

layers = [cl_1, cl_2, pl_2, ml_1, dr_1, ml_2, dr_2, ol_1]

# cross_entropy,mse
loss = cross_entropy
# SGD ,Adagrad,Momentum
optimizer = SGD

# 网络训练，可在train.py中修改你想要的训练过程或者自己编写train函数
a, y = train(train_x, train_y, test_x, test_y, layers, loss, batch_size, eta, epoch, n_sample, interval, optimizer,
             alpha=0.9)
np.save('layer.npy', layers)  # 保存模型

# 混淆矩阵
class1 = np.argmax(a, axis=1).reshape(n_test, 1)
class2 = np.argmax(y, axis=1).reshape(n_test, 1)
c = range(10)
confusion_matrix_plt(class1, class2, c)
