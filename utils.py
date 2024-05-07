import numpy as np

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def keep_ind_only(array, index):
    result = np.zeros_like(array)
    result[index] = array[index]
    
    return result