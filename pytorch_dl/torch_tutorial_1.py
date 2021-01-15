#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch


# In[42]:


# init
'''
# 创建一个5*3的未初始化的Tensor

x = torch.empty(5, 3)
print(x)

# 创建一个5*3 的随机初始化的Tensor
x = torch.rand(5, 3)
print(x)

# 创建一个5*3 的long型全0 的Tensor
x = torch.zeros(5, 3, dtype=torch.long)
x = torch.ones(5,3)
print(x)

# 根据数据创建
x = torch.tensor([5.5, 3])
print(x)

# 使用new_ones，y的属性比如device和dtype是默认继承了另一个张量x的
print(x.dtype)
y = x.new_ones(5, 3, dtype=torch.float64)
print(y,y.dtype)

# 指定新的数据类型
x = torch.randn_like(x, dtype=torch.float)
print(x)
print(x.dtype)
print(x.shape)
# 返回的torch.Size是一个tuple
print(x.size())
'''
print("init")


# In[60]:


# operate
x = torch.rand(5,3)
y = torch.rand(5,3)
'''
# three kinds of addition
print(torch.add(x,y))
result = torch.empty(5, 3)
torch.add(x, y, out = result)
print(result)
# add x to y
y.add_(x)
print(y)
'''
# index访问，索引出来的结果与原数据共享内存
y = x[0, :]
#print(x)
y += 1
#print(x)

# 利用view改变Tensor形状,view返回的tensor与原tensor共享内存
y = x.view(15)
z = x.view(-1, 5)  # -1的维度是可以根据其他维度推导
print(x.size(),y.size(),z.size())
# 若想copy，推荐clone+view
x_cp = x.clone().view(-1,3)
x -= 1
print(x)
print(x_cp)

'''
x = torch.randn(1)
print(x)
# item() 将一个标量Tensor转换成Python number
print(x.item())
'''
# 两个形状不同的Tensor按元素运算时，可能会出发广播broadcasting机制：
# 先适当复制元素使这两个Tensor形状相同后再按元素运算

x = torch.arange(1, 3).view(1, 2)
print(x)
y = torch.arange(1, 4).view(3, 1)
print(y)
print(x+y)

