这是神经网络设计实验的课设

activation_function.py包含了几个激活函数

help_function.py 包含了一些辅助函数，包括卷积中的img2col，函数求导，前向传播，误差计算等几个函数

loss_function.py包含了一些损失函数

network_layer.py包含了几个结构层的定义，卷积，全理解，池化，dropout，以及他们的相关函数，前向计算，反向传播等等

optimizer.py包含了几个优化方法

performance.py包含了混淆矩阵评价指标

train.py 包含了我们设定的训练结构

感知机.py，hebb.py 包含了感知机算法和hebb学习规则

search_example.py 包含了自动参数搜索

三个*example.py文件演示了我们的框架的基础使用教程

本次实验使用fashionmnist进行测试
首先通过数组定义网络结构
然后设定损失函数，激活函数，优化器（也可以参照我们的文件，自己编写相应但函数，然后传入网络训练）
得出结果
网络保存使用np.save直接保存网络层数组。

具体教程课参考*example.py文件，自定义自己的神经网络
