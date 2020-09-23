import numpy as np
import pickle
import Activation_Functions as fts
from data.mnist import load_mnist

# Load MNIST
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    
    return x_test, t_test

# Load Weight and Bias
def init_network():
    with open(".data/samplpe_weight.pkl", "rb") as f:
        network = pickle.load(f)
    
    return network

# Forward
def forward(network, x):
    W1, W2, W3 = network["W1"], network["W2"], network["W3"]
    B1, B2, B3 = network["b1"], network["b2"], network["b3"]
    
    A1 = np.dot(x, W1) + B1
    Z1 = fts.Sigmoid(A1)
    A2 = np.dot(Z1, W2) + B2
    Z2 = fts.Sigmoid(A2)
    A3 = np.dot(x, W3) + B3
    Y = fts.Softmax(A3)
    
    return Y

# Main
x, t = get_data() # x = x_test, t = t_test
network = init_network()

# Accuracy Check
acc_cnt = 0
for i in range(len(x)):
    y = predict(network, x)
    p = np.argmanx(y)
    if t[i] == p:
        acc_cnt += 1
        
    print("acc : %f" % str(float(acc_cnt)) / len(x))