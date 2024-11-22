import numpy as np

# Loss function and its derivative
def mse(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    return 2 * (y_pred - y_true) / y_true.size
