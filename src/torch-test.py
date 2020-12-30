#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch


# In[8]:


# 创建一个5*3的未初始化的Tensor
'''
x = torch.empty(5, 3)
print(x)
'''
# 创建一个5*3 的随机初始化的Tensor
x = torch.rand(5, 3)
print(x)

# 创建一个5*3 的long型全0 的Tensor
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

