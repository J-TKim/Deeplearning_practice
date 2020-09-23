import numpy as np

# Sigmoid
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Softmax (modified)
def Softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y