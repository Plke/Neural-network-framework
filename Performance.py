import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


# -- 混淆矩阵 --
# t为结果向量(n,1)，y为标签向量(n,1)，classes为标签列表(n,)
# 其中默认标签y[i][0]表示classes[y[i][0]]类，比如classes=['猫','狗','鸭子','猪']，若y[i][0]为0则表示为猫
def confusion_matrix_plt(t, y, classes):
    # 构建混淆矩阵
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype='int')
    for i in range(t.shape[0]):
        confusion_matrix[y[i][0]][t[i][0]] += 1
    # print(confusion_matrix)

    # 可视化
    proportion = []
    for i in confusion_matrix:
        for j in i:
            temp = j
            proportion.append(temp)
    # print(np.sum(confusion_matrix[0]))
    # print(proportion)
    # pshow = []
    # for i in proportion:
    #     pt="%.2f%%" % (i * 100)
    #     pshow.append(pt)
    proportion = np.array(proportion).reshape(len(classes), len(classes))  # reshape(列的长度，行的长度)
    # pshow=np.array(pshow).reshape(len(classes),len(classes))
    # print(pshow)
    config = {
        "font.family": 'Times New Roman',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    thresh = confusion_matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(len(classes))] for i in range(len(classes))], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12, color='white',
                     weight=5)  # 显示对应的数字
            # plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12,color='white')
        else:
            plt.text(j, i - 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12)  # 显示对应的数字
            # plt.text(j, i+0.12, pshow[i, j], va='center', ha='center', fontsize=12)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.show()
