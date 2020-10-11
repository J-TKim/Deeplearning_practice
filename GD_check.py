#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from data.mnist import load_mnist
from TwoLayerNet import TwoLayerNet


# In[2]:


# Load data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


# In[3]:


# Define Network
network = TwoLayerNet(input_size=784, hidden_size=5, output_size=10)


# In[4]:


x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network. numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)


# In[5]:


for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))

