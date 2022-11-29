import numpy as np
from help_function import *


# -- 网络层定义 --
class ConvLayer:

    # n_bt:批次尺寸, x_ch:输入的通道数量, x_h:输入图像的高度, x_w:输入图像的宽度
    # n_flt:过滤器的数量, flt_h:过滤器的高度, flt_w:过滤器的宽度
    # stride:步长的幅度, pad:填充的幅度
    # y_ch:输出的通道数量, y_h:输出的高度, y_w:输出的宽度
    # activation:激活函数
    # wb_width:过滤器和偏置的初始值,减小方差
    def __init__(self, x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad , activation,wb_width = 0.1):

        # 将参数集中保存
        self.params = (x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad)


        # 过滤器和偏置的初始值
        self.w = wb_width * np.random.randn(n_flt, x_ch, flt_h, flt_w)
        self.b = wb_width * np.random.randn(1, n_flt)

        # 输出图像的尺寸
        self.y_ch = n_flt  # 输出的通道数量
        self.y_h = (x_h - flt_h + 2*pad) // stride + 1  # 输出的高度
        self.y_w = (x_w - flt_w + 2*pad) // stride + 1  # 输出的宽度

        # AdaGrad算法用
        # self.h_w = np.zeros((n_flt, x_ch, flt_h, flt_w)) + 1e-8
        # if self.bias:
        #     self.h_b = np.zeros((1, n_flt)) + 1e-8

        self.activation=activation

        self.mom_w=np.zeros_like(self.w)

        self.mom_b=np.zeros_like(self.b)

    def forward(self, x, is_train):
        n_bt = x.shape[0]
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        # 将输入图像和过滤器转换成矩阵
        self.cols = im2col(x, flt_h, flt_w, y_h, y_w, stride, pad)
        self.w_col = self.w.reshape(n_flt, x_ch*flt_h*flt_w)

        # 输出的计算：矩阵乘积、偏置的加法运算、激活函数

        u = np.dot(self.w_col, self.cols).T + self.b


        z=u.reshape(n_bt, y_h, y_w, y_ch).transpose(0, 3, 1, 2)
        a=self.activation(z)
        return a,z

    def backward(self, grad_y,z):
        n_bt = grad_y.shape[0]
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w


        delta = grad_y *derivativ(self.activation,z,1e-6)
        delta = delta.transpose(0,2,3,1).reshape(n_bt*y_h*y_w, y_ch)

        # 过滤器和偏置的梯度
        grad_w = np.dot(self.cols, delta)
        grad_w = grad_w.T.reshape(n_flt, x_ch, flt_h, flt_w)

        grad_b = np.sum(delta, axis=0)

        # 输入的梯度
        grad_cols = np.dot(delta, self.w_col)
        x_shape = (n_bt, x_ch, x_h, x_w)
        grad_x = col2im(grad_cols.T, x_shape, flt_h, flt_w, y_h, y_w, stride, pad)


        return grad_x,grad_w,grad_b


    def update(self, eta,grad_w,optimizer,grad_b=None,alpha=0.9):
        # self.h_w += grad_w * grad_w
        # self.w -= eta / np.sqrt(self.h_w) * grad_w

        # self.h_b += grad_b * grad_b
        # self.b -= eta / np.sqrt(self.h_b) * grad_b

        self.w,self.b,self.mom_w,self.mom_b=optimizer(eta,self.mom_w,grad_w,self.w,grad_b,self.mom_b,self.b,alpha=0.9)

    #重置网络可学习参数，结构不变
    def remake(self,wb_width=0.1):
        x_ch, x_h, x_w, n_flt, flt_h, flt_w, stride, pad = self.params
        self.w = wb_width * np.random.randn(n_flt, x_ch, flt_h, flt_w)
        self.b = wb_width * np.random.randn(1, n_flt)


# -- 池化层 --
class PoolingLayer:

    # n_bt:批次尺寸, x_ch:输入的通道数量, x_h:输入图像的高度, x_w:输入图像的宽度
    # pool:池化区域的尺寸, pad:填充的幅度
    # y_ch:输出的通道数量, y_h:输出的高度, y_w:输出的宽度

    def __init__(self, x_ch, x_h, x_w, pool, pad):

        # 将参数集中保存
        self.params = (x_ch, x_h, x_w, pool, pad)

        self.w=None
        self.b=None
        # 输出图像的尺寸
        self.y_ch = x_ch  # 输出的通道数量
        self.y_h = x_h//pool if x_h%pool==0 else x_h//pool+1  # 输出的高度
        self.y_w = x_w//pool if x_w%pool==0 else x_w//pool+1  # 输出的宽度

    def forward(self, x, is_train):
        n_bt = x.shape[0]
        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        # 将输入图像转换成矩阵
        cols = im2col(x, pool, pool, y_h, y_w, pool, pad)
        cols = cols.T.reshape(n_bt*y_h*y_w*x_ch, pool*pool)

        # 输出的计算：最大池化
        y = np.max(cols, axis=1)
        z = y.reshape(n_bt, y_h, y_w, x_ch).transpose(0, 3, 1, 2)

        # 保存最大值的索引值
        self.max_index = np.argmax(cols, axis=1)

        return z,z

    def backward(self, grad_y,z):
        n_bt = grad_y.shape[0]

        x_ch, x_h, x_w, pool, pad = self.params
        y_ch, y_h, y_w = self.y_ch, self.y_h, self.y_w

        grad_y = grad_y.reshape(n_bt, y_ch, y_h, y_w)

        # 对输出的梯度的坐标轴进行切换
        grad_y = grad_y.transpose(0, 2, 3, 1)

        # 创建新的矩阵，只对每个列中具有最大值的元素所处位置中放入输出的梯度
        grad_cols = np.zeros((pool*pool, grad_y.size))
        grad_cols[self.max_index.reshape(-1), np.arange(grad_y.size)] = grad_y.reshape(-1)
        grad_cols = grad_cols.reshape(pool, pool, n_bt, y_h, y_w, y_ch)
        grad_cols = grad_cols.transpose(5,0,1,2,3,4)
        grad_cols = grad_cols.reshape( y_ch*pool*pool, n_bt*y_h*y_w)

        # 输入的梯度
        x_shape = (n_bt, x_ch, x_h, x_w)
        grad_x = col2im(grad_cols, x_shape, pool, pool, y_h, y_w, pool, pad)

        return grad_x,None,None

    def update(self, eta,grad_w,optimizer,grad_b=None,alpha=0.9):
        pass
    def remake(self,wb_width=0.1):
        pass


# -- 全链接层的祖先类 --
class BaseLayer:
    # n_upper:输入维度, n:输出维度
    # activation:激活函数
    # wb_width:过滤器和偏置的初始值,减小方差
    def __init__(self, n_upper, n,activation,wb_width=0.1):
        self.w = wb_width * np.random.randn(n_upper, n)

        self.b = wb_width * np.random.randn(n)

        # self.h_w = np.zeros(( n_upper, n)) + 1e-8
        # if bias:
        #     self.h_b = np.zeros(n) + 1e-8

        self.activation=activation
        self.i=n_upper
        self.o=n
        self.mom_w=np.zeros_like(self.w)

        self.mom_b=np.zeros_like(self.b)

    def update(self, eta,grad_w,optimizer,grad_b=None,alpha=0.9):
        # self.h_w += grad_w * grad_w
        # self.w -= eta / np.sqrt(self.h_w) * grad_w

        # self.h_b += grad_b * grad_b
        # self.b -= eta / np.sqrt(self.h_b) * grad_b

        self.w,self.b,self.mom_w,self.mom_b=optimizer(eta,self.mom_w,grad_w,self.w,grad_b,self.mom_b,self.b,alpha=0.9)

        # h_w=np.zeros_like(grad_w)+1e-8
        # h_b=np.zeros_like(grad_b)+1e-8
        # if self.h_b.all()!=h_b.all():
        #     print(1515485848)

        # h_w += grad_w * grad_w
        # self.w -= eta / np.sqrt(h_w) * grad_w

        # h_b += grad_b * grad_b
        # self.b -= eta / np.sqrt(h_b) * grad_b
    def remake(self,wb_width=0.1):
        n_upper=self.i
        n=self.o
        self.w = wb_width * np.random.randn(n_upper, n)
        self.b = wb_width * np.random.randn(n)


# -- 全链接的中间层 --
class MiddleLayer(BaseLayer):
    def forward(self, x, is_train):
        n_bt = x.shape[0]
        self.x = x.reshape(n_bt, -1)

        z = np.dot(self.x, self.w) + self.b


        a=self.activation(z)
        return a,z

    def backward(self, grad_y,z):

        delta = grad_y * derivativ(self.activation,z,1e-6)

        grad_w = np.dot(self.x.T, delta)

        grad_b = np.sum(delta, axis=0)

        grad_x = np.dot(delta, self.w.T)


        return grad_x,grad_w,grad_b


# -- 全链接的输出层 --
class OutputLayer(BaseLayer):
    def forward(self, x, is_train):
        n_bt = x.shape[0]
        self.x = x.reshape(n_bt, -1)

        z = np.dot(self.x, self.w) + self.b


        a=self.activation(z)
        return a,z

    def backward(self, a,t,z,loss):#t:实际，a：预测 ,z :激活前

        delta=np.matmul(Jacobian(self.activation,z,1e-6),derivativ2(loss,a,t,1e-6)[:,:,np.newaxis] )
        # print(delta.shape)
        delta=np.squeeze(delta,axis=2)

        grad_w = np.dot(self.x.T, delta)

        grad_b = np.sum(delta, axis=0)

        grad_x = np.dot(delta, self.w.T)

        return grad_x,grad_w,grad_b


# -- Dropout --
class Dropout:
    def __init__(self, dropout_ratio):
        self.dropout_ratio = dropout_ratio
        self.w=None
        self.b=None

    def forward(self, x, is_train):
        if is_train:
            rand = np.random.rand(*x.shape)
            self.dropout = np.where(rand > self.dropout_ratio, 1, 0)
            z = x * self.dropout
        else:
            z = (1-self.dropout_ratio)*x
        return z,z

    def backward(self, grad_y,z):
        grad_x = grad_y * self.dropout
        return grad_x,None,None
    def update(self, eta,grad_w,optimizer,grad_b=None,alpha=0.9):
        pass
    def remake(self,wb_width=0.1):
        pass