#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import packages
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.optimizer import *
from collections import OrderedDict


# In[2]:


# Define a function
def f(x, y):
    return x**2 / 20.0 + y**2
    
def df(x, y):
    return x/10.0, 2.0*y


# In[3]:


# Initalize
x0 = (-7.0, 2.0)
params = {}
params["x"], params["y"] = x0[0], x0[1]

grads = {}
grads["x"], grads["y"] = 0, 0


# In[11]:


# Optimizers
optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=0.5)
optimizers["RMSprop"] = RMSprop(lr=0.15)


# In[13]:


idx = 1
# Main iteration
for key in optimizers:
    optimizer = optimizers[key]
    x_history = []
    y_history = []
    params["x"], params["y"] = x0[0], x0[1]
    
    for i in range(100):
        x_history.append(params["x"])
        y_history.append(params["y"])
        grads["x"], grads["y"] = df(params["x"], params["y"])
        optimizer.update(params, grads)
    
    # Visualize
    x = np.arange(-10, 10, 0.01)
    y = np.arange(-5, 5, 0.01)

    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    
    # Simplify for contour
    mask = Z > 7
    Z[mask] = 0
    
    # Graph
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(x_history, y_history, "o-", color="red")
    plt.contour(X, Y, Z)
    
plt.show()

