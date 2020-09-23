#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from data.mnist import load_mnist
from PIL import Image
import matplotlib.pyplot as plt


# In[2]:


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# In[3]:


# Load MNIST
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=False)


# In[4]:


img = x_train[0]
label = y_train[0]
print(label)


# In[5]:


print(img.shape)


# In[6]:


img = img.reshape(28, 28)
print(img.shape)


# In[7]:


plt.imshow(img)

