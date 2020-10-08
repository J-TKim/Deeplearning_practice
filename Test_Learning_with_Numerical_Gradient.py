#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Load packages
import numpy as np
import matplotlib.pyplot as plt
from data.mnist import load_mnist
from TwoLayerNet import TwoLayerNet


# In[2]:


# MNIST Loader
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)


# In[3]:


# Hyperparameter
train_loss_list = []
iter_list = []
iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_late = 0.1


# In[4]:


# Define TwoLayerNet
network =TwoLayerNet(input_size=784, hidden_size=50, output_size=10)


# In[ ]:


# Mini-batch Learning
for i in range(iters_num):
    # For Debugging
    if ((i+1)%10) == 0:
        print("iter :", i+1)
    # Mini-batch setup
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # Compute Gradient
    grad = network.numerical_gradient(x_batch, t_batch)
    
    # SGD
    for key in ("W1", "b1", "W2", "b2"):
        network.params[key] -= learning_late * grad[key]
        
    # Record the history
    loss = network.loss(x_batch, t_batch)
    iter_list.append(i+1)
    train_loss_list.append(loss)


# In[ ]:


plt.plot(iter_list, train_loss_list)
plt.showw()

