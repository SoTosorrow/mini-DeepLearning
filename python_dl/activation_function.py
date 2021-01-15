import numpy as np
import matplotlib.pylab as plt

# activation function
# 朴素感知机（单层网络）中，激活函数使用了阶跃函数的模型
# 阶跃函数是指一旦输入超过阈值，就切换输出的函数
# 多层感知机使用的是sigmoid等平滑的激活函数

# 阶跃函数
def step_function(x):
    #if x > 0:
    #    return 1
    #else:
    #    return 0
    
    #y = x >0
    #return y.astype(np.int)

    
    return np.array(x>0, dtype=np.int)    

# sigmoid function
# h(x) = 1/ (1+ exp(-x))
def sigmoid(x):
    return 1/(1+np.exp(-x))

# relu(Rectified Linear Unit)
# 输入>0时直接输出该值，输入<=0时输出0
def relu(x):
    # 从输入的数值中选择较大的值输出
    return np.maximum(0,x)

# softmax
# softmax的受到前一层所有输入信号影响
# 分子是输入信号ak的指数函数，分母是所有输入信号的指数函数和
def softmax_primer(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
# 上述softmax有溢出问题
# 改进
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

if __name__ == '__main__':
    x = np.arange(-5.0,5.0,0.1)
    # step
    y = step_function(x)
    plt.plot(x,y)
    plt.ylim(-0.1,1.1)
    plt.show()
    # sigmoid
    y = sigmoid(x)
    plt.plot(x,y)
    plt.ylim(-0.1,1.1)
    plt.show()
    # relu
    y = relu(x)
    plt.plot(x,y)
    plt.ylim(-0.1,1.1)
    plt.show()
    # softmax
    y = softmax(x)
    plt.plot(x,y)
    plt.ylim(-0.1,1.1)
    plt.show()
