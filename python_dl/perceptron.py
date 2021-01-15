import numpy as np

# perceptron process AND
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    # theta is also the bias
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        # activate
        return 1


def AND_np(x1, x2):
    x = np.array([x1, x2])# input
    w = np.array([0.5, 0.5]) # weight
    b = -0.7 # bias (the easyness of net active)
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])# input
    w = np.array([-0.5, -0.5]) # weight
    b = 0.7 # bias (the easyness of net active)
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1

# 单层感知机无法分离非线性空间
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y



if __name__ == '__main__':
    print(XOR(0,0),
            XOR(1,0),
            XOR(0,1),
            XOR(1,1))




    


