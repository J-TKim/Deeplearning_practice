import numpy as np

# Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-2))

# Softmax (Modified)
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp - a / sum - exp