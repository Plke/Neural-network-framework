from help_function import *
import matplotlib.pyplot as plt
from loss_function import *
from  activation_function import *
from network_layer import *
from  Performance import *
from optimizer import *


# train_x: 训练数据  train_y：训练标签 test_x：测试数据  test_y：测试标签
# layers：定义的网络 loss：损失函数  训练批大小：batch_size 学习率：eta
# 训练次数：epoch 误差计算样本数：n_sample interval 显示间隔：interval
# optimizer: 优化器   alpha：alpha
def train(train_x, train_y, test_x, test_y, layers, loss, batch_size=100, eta=0.001, epoch=50, n_sample=200,
          interval=10, optimizer=Adagrad, alpha=0.9):
    n_train = train_x.shape[0]  # 训练数据的采样数
    n_test = test_x.shape[0]  # 测试数据的采样数
    n_batch = n_train // batch_size
    train_error = []
    test_error = []
    for i in range(epoch):
        # -- 误差的测算 --
        # 在训练过程中选取部分样本进行误差计算
        x, y, a = forward_sample(layers, train_x, train_y, n_sample)
        error_train = get_error(a, y, n_sample, loss)

        x, y, a = forward_sample(layers, test_x, test_y, n_sample)
        error_test = get_error(a, y, n_sample, loss)

        train_error.append(error_train)
        test_error.append(error_test)

        # -- 处理进度的显示 --
        if interval != 0 and i % interval == 0:
            print("Epoch:" + str(i) + "/" + str(epoch),
                  "  Error_train:" + str(error_train),
                  "  Error_test:" + str(error_test))

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

    # -- 显示记录误差的表格 --

    plt.plot(train_error, label="Train")
    plt.plot(test_error, label="Test")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

    # -- 正确率的测定 --
    x, y, a = forward_sample(layers, train_x, train_y, n_train)
    count_train = np.sum(np.argmax(a, axis=1) == np.argmax(y, axis=1))

    x, y, a = forward_sample(layers, test_x, test_y, n_test)
    count_test = np.sum(np.argmax(a, axis=1) == np.argmax(y, axis=1))

    if interval != 0:
        print("Accuracy Train:", str(count_train / n_train * 100) + "%",
              "Accuracy Test:", str(count_test / n_test * 100) + "%")
    return a, y
