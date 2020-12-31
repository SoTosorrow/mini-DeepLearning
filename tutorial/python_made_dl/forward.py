import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

# 一般地，输出层使用的激活函数
# 回归问题可以使用恒等函数
# 二元分类可以使用sigmoid函数
# 多元分类可以使用softmax函数

# 恒等函数
def identity_function(x):
    return x

# 将神经网络运算作为矩阵运算打包进行
'''
if __name__ == '__main__':
    # 2*(2,3)=3 
    print("first")
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])# 将偏置作为前一层值为1的神经元

    print(W1.shape) # (2,3)
    print(X.shape) # (2,)
    print(B1.shape) # (3,)

    # np.dot()
    A1 = np.dot(X,W1) + B1
    Z1 = sigmoid(A1)
    
    print(A1)
    print(Z1)

    print("second")
    W2 = np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])
    B2 = np.array([0.1, 0.2]) # 第二层偏置神经元
    
    print(Z1.shape) # (3,)
    print(W2.shape) # (3,2)
    print(B2.shape) # (2,)

    # 第一层输出Z1变成第二层输入
    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)
'''
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3],[0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

# 前向：输入到输出方向的传递处理
def forward(network, x):
    W1, W2, W3 = network['W1'],network['W2'],network['W3']
    b1, b2, b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network,x)
print(y)





