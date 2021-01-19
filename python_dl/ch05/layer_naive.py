# coding: utf-8
# 计算图的简单实现例子

# 实现乘法层
class MulLayer:
    # 初始化实例变量xy 用于保存正向传播时的输入
    def __init__(self):
        self.x = None
        self.y = None

    # 正向传播
    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    # 反向传播，传播的是该节点的导数
    # 将后面网络传来的导数dout乘以正向传播的翻转值再传给前层
    def backward(self, dout):
        dx = dout * self.y # 翻转x y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    # 加法导数为1
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
