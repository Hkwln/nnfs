import numpy as np

# activation function and its deriviative
def tanh(x):
    return np.tanh(x)
def tanh_prime(x):
    return 1-np.tanh(x)**2

