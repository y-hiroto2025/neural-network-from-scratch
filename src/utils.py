import numpy as np

def function(x):
    return x**4 + x

def numerical_diff(f, x):
    """
    数値微分(中心差分)
    """
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)



x = 1

print(numerical_diff(function, x))