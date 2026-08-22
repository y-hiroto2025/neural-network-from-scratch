import numpy as np

def function_1(x):
    return x**2

def function_2(x):
    return x[0]**2 + x[1]**2

def function_2_tmp(x_0):
    """
    x[1] = 4.0 のとき
    """
    return x_0**2 + 4.0**2

def numerical_diff(f, x):
    """
    数値微分(中心差分)
    """
    h = 1e-5
    return (f(x+h) - f(x-h)) / (2*h)

print(f"微分: x**2 => {numerical_diff(function_1, 1)}")