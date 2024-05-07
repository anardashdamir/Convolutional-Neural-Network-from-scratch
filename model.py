import numpy as np
from utils import cross_entropy_loss

import time
from tqdm import tqdm





class Network:
    def __init__(self, layers, optimizer=None):
        self.layers = layers
        self.optimizer = optimizer

    def add(self, layer):
        self.layers.append(layer)
    
    def fit(self, features, labels, epochs):
        

        for epoch in range(1, epochs+1):

            total_loss = 0
            training_accuracy = 0
            time_start = time.time()

            for out, label in tqdm(zip(features, labels), total=len(features)):
                # FORWARD
                for layer in self.layers:
                    out = layer(out)
                    

                #BACKWARD
                grad_a =  - label / (self.layers[-1].a + 1e-8) + (1 - label)/(1 - self.layers[-1].a + 1e-8)
                
                for layer in reversed(self.layers):
                    if layer.trainable:
                        
                        grad_a, grad_params, params = layer.backward(grad_a)
                        self.optimizer.update_parameters(layer, params, grad_params)
                        
                    else:
                        grad_a = layer.backward(grad_a)

                # calculate loss
                y_pred = 1 if out >= 0.5 else 0
                if y_pred == label:
                    training_accuracy += 1    
                loss = cross_entropy_loss(label, y_pred)
                total_loss += loss
                
            cost = total_loss / len(features)
            time_end = time.time()
                
            print(f"Epoch {epoch }/{epochs} | Loss: {cost:.4f} | Training Accuracy: {training_accuracy / len(features):.4f} | Time Spent:{time_end - time_start:.2f}s")
        
