import numpy as np

def hardlims(x):
    return np.where(x<0,-1,1)

# 无监督Hebb学习算法
# x的每个列向量表示一个样本，alpha表示学习率，epho表示次数，func表示激活函数可进行传参
# 原理及测试样例来源：https://blog.csdn.net/qq_17517409/article/details/106027837
def unsupervised_Hebb_learning(x,alpha,epho,func,**kwargs):
    w=np.random.randn(x.shape[0],1)
    # w=np.array([[1,-1,0,0.5]]).T
    for i in range(epho):
        for j in range(x.shape[1]):
            res=np.dot(w.T,x[:,j].T)
            a=func(res)
            b=alpha*a*np.array([x[:,j]]).T
            w=w+b
    result=func(np.dot(w.T,x))
    return w,result

'''
# 无监督测试样例
x=np.array([[1,-2,1.5,0],[1,-0.5,-2,-1.5],[0,1,-1,1.5]]).T
alpha=0.5
epho=100
w,result=unsupervised_Hebb_learning(x,alpha,epho,func=hardlims)
print(w)
print(result)
'''

# 有监督的Hebb学习算法
# x的每个列向量表示一个样本，维度为(a,n);y的每个列向量表示一个标签，维度为(b,n);test为测试样本
# 原理来源书本P92
def supervised_Hebb_learning(x,y,test,alpha,epho,func,**kwargs):
    # w=np.random.randn(y.shape[0],x.shape[0])
    w=np.zeros((y.shape[0],x.shape[0]))
    for i in range(epho):
        w=w+alpha*np.dot(y,x.T)
    result=func(np.dot(w,test))
    return w,result

# 有监督测试样例
# 测试样例来源书本P102
x=np.array([[1,-1,1,1],[1,1,-1,1],[-1,-1,-1,1]]).T
y=np.array([[-1,-1],[-1,1],[1,-1]]).T
test=np.array([[1,-1,1,-1]]).T
alpha=1
epho=1
w,result=supervised_Hebb_learning(x,y,test,alpha,epho,func=hardlims)
print(w)
print(result)