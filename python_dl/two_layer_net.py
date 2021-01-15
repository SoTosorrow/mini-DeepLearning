import sys,os
sys.path.append(os.pardir)
from common.functions import *
from common.gradient import numerical_gradient

class TwoLayerNet:

    # 输入层神经元数、隐藏层神经元数、输出层神经元数
    def __init__(self,input_size,hidden_size,output_size,
                weight_init_std=0.01):
        # 初始化权重,params保存神经网络的参数的字典型变量
        # 初始化权重参数，使用符合高斯分布的随机数初始化（偏置使用0 初始）
        # W为层权重 b为层偏置
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.rand(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * \
                            np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self,x):
        W1, W2 = self.params['W1'],self.params['W2']
        b1, b2 = self.params['b1'],self.params['b2']
        
        # 两层神经网络，激活函数分别位sigmoid,softmax
        a1 = np.dot(x,W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,W2) + b2
        y = softmax(a2)

        return y


    # 计算损失函数、输入数据x、监督数据t
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y,t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        y = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # 计算权重参数的梯度、输入数据x、监督数据t
    def numerical_gradient(self,x,t):
        loss_W = lambda W:self.loss(x, t)
        # 保存梯度的字典型变量(numerical_gradient方法返回值)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    # numerical_gradient 高速版
    def gradient(self, x, t):
        pass


x = np.random.rand(100,784) # shape = (100,784)
t = x +1
'''
print('t:',t)
net = TwoLayerNet(784,100,10)
print('t2:',t)
y = net.predict(x)
print('t3:',t)

grads = net.numerical_gradient(x, t)


print(grads['W1'].shape)

print(grads['b2'].shape)
'''
