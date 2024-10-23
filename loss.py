import numpy as np
#folgendes hat der debugger vorgeschlagen
#following import is needed to use numpy functions in tensorflow
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
#loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size
