import numpy as np


# -- im2col --
def im2col(images, flt_h, flt_w, out_h, out_w, stride, pad):

    n_bt, n_ch, img_h, img_w = images.shape

    img_pad = np.pad(images, [(0,0), (0,0), (pad, pad), (pad, pad)], "constant")
    cols = np.zeros((n_bt, n_ch, flt_h, flt_w, out_h, out_w))

    for h in range(flt_h):
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w
            cols[:, :, h, w, :, :] = img_pad[:, :, h:h_lim:stride, w:w_lim:stride]

    cols = cols.transpose(1, 2, 3, 0, 4, 5).reshape(n_ch*flt_h*flt_w, n_bt*out_h*out_w)
    return cols

# -- col2im --
def col2im(cols, img_shape, flt_h, flt_w, out_h, out_w, stride, pad):

    n_bt, n_ch, img_h, img_w = img_shape

    cols = cols.reshape(n_ch, flt_h, flt_w, n_bt, out_h, out_w, ).transpose(3, 0, 1, 2, 4, 5)
    images = np.zeros((n_bt, n_ch, img_h+2*pad+stride-1, img_w+2*pad+stride-1))

    for h in range(flt_h):
        h_lim = h + stride*out_h
        for w in range(flt_w):
            w_lim = w + stride*out_w
            images[:, :, h:h_lim:stride, w:w_lim:stride] += cols[:, :, h, w, :, :]

    return images[:, :, pad:img_h+pad, pad:img_w+pad]

#函数求导，函数 f 对变量 x 求导
def derivativ(f, x, dx,usediff=False):
    if not usediff:
        y1 = f(x-dx)
        y2 = f(x + dx)
        return (y2 - y1) / (dx*2)
    else:
        den=x.shape
        x=x.reshape(x.shape[0],-1)
        m,n=x.shape
        d=dx*np.identity(n)
        x=np.expand_dims(x,axis=1).repeat(n,axis=1)
        x1=x+d
        x2=x-d
        return np.diagonal((f(x1)-f(x2))/(2*dx),axis1=1,axis2=2)
#损失函数求导，函数 f 对变量 x 求偏导
def derivativ2(f, x,a, dx):
    y1 = f(x-dx,a)
    y2 = f(x + dx,a)
    return (y2 - y1) / (dx*2)

def Jacobian(f, x, dx):
    den=x.shape
    x=x.reshape(x.shape[0],-1)
    m,n=x.shape
    d=dx*np.identity(n)
    x=np.expand_dims(x,axis=1).repeat(n,axis=1)
    x1=x+d
    x2=x-d
    return ((f(x1)-f(x2))/(2*dx))


# -- 正向传播 --
def forward_propagation(layers,a, is_train):
    z={}
    for i in range(len(layers)):
        a,z[i]=layers[i].forward(a,is_train)

    return a,z


# -- 反向传播 --
def backpropagation(layers,t,a,z,loss,optimizer=None): #t:实际，a：预测
    #assert()
    grad_w={}
    grad_b={}
    L=len(layers)

    grad_x,grad_w[L-1],grad_b[L-1] =layers[-1].backward(a,t,z[L-1],loss)


    for i in range(L-2,-1,-1):

        grad_x,grad_w[i],grad_b[i]=layers[i].backward(grad_x,z[i])


    return grad_w,grad_b



# -- 权重和偏置的更新 --
def uppdate_wb(layers,grad_w,eta,optimizer,grad_b=None,alpha=0.9):

    for i in range(len(layers)):

        layers[i].update(eta,grad_w[i],optimizer,grad_b[i],alpha=0.9)




# -- 对误差进行计算 --
def get_error(a,y, batch_size,loss):
    return np.sum(loss(a,y)) / batch_size

# -- 对样本进行正向传播 --
def forward_sample(layers,input, label, n_sample):
    index_rand = np.arange(len(label))
    np.random.shuffle(index_rand)
    index_rand = index_rand[:n_sample]
    x = input[index_rand, :]
    y = label[index_rand, :]
    a,z=forward_propagation(layers,x, False)
    return x, y,a

#重置模型，只改变网络的可学习参数，不改变网络结构
def remake_model(layers,wb_width=0.1):
    for layer in layers:
        layer.remake(wb_width)
    return layers