import numpy as np
import matplotlib.pylab as plt
import sys,os

sys.path.append(os.pardir) # 为了导入父目录中的文件而进行的设定
from dataset.mnist import load_mnist
from PIL import Image
import pickle
from activation_function import *

def img_show(img):
    # numpy to PIL: Image.fromarray()
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
'''
normalize=True 是否将输入图像正规化为0-1
flatten 设置是否展开输入图像（变成一维数组）1*28*28=784
one_hot_label 是否将标签保存为one-hot。
one-hot为仅有正确解标签为1其余为0的数组

(x_train, t_train),(x_test, t_test) = load_mnist(flatten=True,
        normalize=False)

print(x_train.shape) # (60000,784)
print(t_train.shape)
print(x_test.shape) # (10000,784)
print(t_test.shape)


img = x_train[0]
label = t_train[0]
img = img.reshape(28,28) # flatten 784 reshape to 28*28
img_show(img)
'''

def get_data():
    (x_train,t_train),(x_test,t_test) = load_mnist(
            normalize=True,flatten=True,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("./deep_learning_demo/ch03/sample_weight.pkl","rb")as f:
        network = pickle.load(f)
    return network

def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y

# get test data
x,t = get_data()
# the network trained
network = init_network()
# obs
W1,W2,W3 = network['W1'],network['W2'],network['W3']
print(x.shape,x[0].shape)
print(W1.shape,W2.shape,W3.shape)


# single process
'''
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network,x[i])
    # np.argmax(x)函数取出数组中的最大值的索引
    p = np.argmax(y)  # get the index of the most likely data
    # predict right
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:"+str(float(accuracy_cnt)/len(x)))
'''

# batch process
batch_size = 100  # 一次输入100
accuracy_cnt = 0
for i in range(0,len(x),batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p==t[i:i+batch_size])
print("Accuracy:"+str(float(accuracy_cnt)/len(x)))



