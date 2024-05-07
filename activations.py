import numpy as np


def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

    
def sigmoid_derivative(x):
    _sigmoid = sigmoid(x)
    return _sigmoid * (1 - _sigmoid)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)


activations = {
    'relu': relu,
    'sigmoid': sigmoid,
}

activation_derivatives = {
    'relu': relu_derivative,
    'sigmoid': sigmoid_derivative
}