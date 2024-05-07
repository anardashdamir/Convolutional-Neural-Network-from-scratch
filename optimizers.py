import numpy as np


class SGD:
    def __init__(self, lr = 0.001):
        self.lr = lr
        
    def update_parameters(self, layer, parameters, param_gradients):
        
        for param, grad in zip(parameters, param_gradients):
            param -= self.lr * grad
    

class Adam:
    def __init__(self, lr = 0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8

    def update_parameters(self, layer, parameters, param_gradients):
        if not hasattr(layer, 'v'):
            layer.v = [np.zeros_like(p) for p in parameters]
            layer.s = [np.zeros_like(p) for p in parameters]
                        
        for i, (param, grad) in enumerate(zip(parameters, param_gradients)):
            

            layer.v[i] = self.beta1 * layer.v[i] + (1 - self.beta1) * grad
            layer.s[i] = self.beta1 * layer.s[i] + (1 - self.beta1) * grad**2
            
            
            param -= self.lr * layer.v[i] / (np.sqrt(layer.s[i]) + self.epsilon)