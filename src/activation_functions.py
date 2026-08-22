import numpy as np

def step_function(x):
    y = x > 0
    return y.astype(int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def ReLU(x):
    return np.maximum(0, x)

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a
    return y

def Tanh(x):
    return np.tanh(x)