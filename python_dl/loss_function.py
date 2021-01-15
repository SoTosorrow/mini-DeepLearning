import numpy as np

# loss function 
# 表示神经网络性能（拟合）

# mean squared error 均方误差
def mean_squared_error(y, t):
    # y is the output of network, t is the train_data
    return 0.5 * np.sum((y-t)**2)

# croess_entropy error 交叉熵误差
def cross_entropy_error(y, t):
    # 当出现log(0)时，np.log(0)会变为负无限大的-inf，为避免而添加一个微小值delta
    delta = 1e-7
    return -np.sum(t* np.log(y + delta))




