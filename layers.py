import numpy as np
from activations import activations, activation_derivatives
from utils import keep_ind_only


class Conv:

    def __init__(self, n_filters=1, filter_size=(3, 3), padding=0, stride=1, trainable=True):
        self.name = "conv"

        self.n_filters = n_filters
        self.padding = padding
        self.stride = stride
        self.filter_size = filter_size
        self.filters = None
        self.trainable = trainable


    def forward(self, img):
   
        self.channel_size = img.shape[2] if len(img.shape) == 3 else 1
            
        if self.filters is None:    
            self.filters = np.random.uniform(-1, 1, size=(self.n_filters, *self.filter_size, self.channel_size)) 


        # SET OUTPUT FILTER
        out_height = ((img.shape[0] - self.filter_size[0] + 2 * self.padding) / self.stride) + 1
        out_width = ((img.shape[1] - self.filter_size[1] + 2 * self.padding) / self.stride) + 1
        
        
        # STRIDE CHECK
        if int(out_height) != out_height or int(out_width) != out_width:
  
            raise  Exception(f'Stride {self.stride} is incorrect')
            
            
        out = np.zeros((int(out_height), int(out_width), self.n_filters))
        
    

        if self.padding:
            img = np.pad(
                img,
                ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                mode="constant",
            )

        self.inp_last = img

        # CONV  
        for f in range(self.n_filters):
            for out_col, col in enumerate(range(0, img.shape[1] - self.filter_size[1] + 1, self.stride)):
                for out_row, row in enumerate(range(0, img.shape[0] - self.filter_size[0] + 1,self.stride)):
                
                    slice = img[row : self.filter_size[0] + row, 
                                col : self.filter_size[1] + col,]

                    out[out_row, out_col, f] = np.sum(slice * self.filters[f])

            out += out
                
        return out

    def backward(self, dl_da):
        
        # RESHAPE gradiet as filter
        dl_da = dl_da.transpose(2, 0, 1)
        dl_da = dl_da[..., np.newaxis]
        dl_da = np.tile(dl_da, (1, 1, 1, self.channel_size))
        
        # SET output for previous layer and gradient of filter
        grad_a = np.zeros((self.n_filters, *self.inp_last.shape))
        grad_f = np.zeros_like(self.filters)
        
        


        # Update Filter
        for f in range(self.n_filters):
            for out_col, col in enumerate(range(0, self.inp_last.shape[1] - dl_da.shape[2] + 1, self.stride)):
                for out_row, row in enumerate(range(0, self.inp_last.shape[0] - dl_da.shape[1] + 1, self.stride)):
                
                    slice = self.inp_last[row : dl_da.shape[1] + row, 
                                          col : dl_da.shape[2] + col,]

                    grad_f[f, out_row, out_col] = np.sum(slice * dl_da[f])
                
                
        # gradient for input      
        filter = self.filters
        padding = dl_da.shape[1] - 1
        pad_width = ((padding, padding), (padding, padding), (0, 0))
        
        for f in range(self.n_filters):
            
            padded_filter = np.pad(np.flip(filter[f], 0), pad_width, mode="constant", constant_values=0)
            
            for out_col, col in enumerate(range(0, padded_filter.shape[1] - dl_da.shape[2] + 1, self.stride)):
                for out_row, row in enumerate(range(0, padded_filter.shape[0] - dl_da.shape[1] + 1,self.stride)):
                
                    slice = padded_filter[row : dl_da.shape[1] + row, 
                                          col : dl_da.shape[2] + col,]

                    grad_a[f, out_row, out_col] = np.sum(slice * dl_da[f])
                    
        grad_a = np.mean(grad_a, axis=0)
    

        return grad_a if not self.trainable else (grad_a, (grad_f), (self.filters))

    __call__ = forward
    

# MAX Pooling layer 

class MaxPool:
    def __init__(self, filter_size=3, stride=1):
        self.filter_size = filter_size
        self.stride = stride
        self.trainable = None
    
    def forward(self, inp):
        
        self.inp_last = inp

        out_size = ((inp.shape[0] - self.filter_size) / self.stride) + 1
        
        if int(out_size) != out_size :
            raise  Exception(f'Stride {self.stride} is incorrect')
        else:
            out_size = int(out_size)
        
        out = np.zeros((out_size, out_size, inp.shape[2]))
        
        for channel in range(inp.shape[2]):
            for out_col, col in enumerate(range(0, inp.shape[1] - self.filter_size + 1, self.stride)):
                for out_row, row in enumerate(range(0, inp.shape[0] - self.filter_size + 1,self.stride)):
                
                    slice = inp[row : self.filter_size + row, 
                                col : self.filter_size + col,
                                channel]

                    out[out_row, out_col, channel] = np.max(slice)
                    
        
        return out
    
    def backward(self, dl_da):
        
        grad_a = np.zeros_like(self.inp_last)
        
        mask = self.inp_last
        

        for channel in range(self.inp_last.shape[2]):
            for out_col, col in enumerate(range(0, self.inp_last.shape[1] - dl_da.shape[1] + 1, self.stride)):
                for out_row, row in enumerate(range(0, self.inp_last.shape[0] - dl_da.shape[0] + 1,self.stride)):
                
                    slice = self.inp_last[row : dl_da.shape[0] + row, 
                                          col : dl_da.shape[1] + col,
                                          channel].copy()
                    
                    max_ind = np.unravel_index(np.argmax(slice), slice.shape)
                    slice = keep_ind_only(slice, max_ind)
                    
                    grad_a[row : dl_da.shape[0] + row, 
                           col : dl_da.shape[1] + col,
                           channel]                      += slice * dl_da[:, :, channel]
        return grad_a
    
    __call__ = forward



# ReLU activation

class ReLU:
    def __init__(self, name='relu'):
        self.name = name
        self.trainable = None

    def forward(self, inp):
        self.inp = inp
        
        
        out = activations['relu'](inp)
        return out

    def backward(self, dl_da):
        
        grad_a = dl_da * activation_derivatives['relu'](self.inp)
        
        return grad_a
    
    __call__ = forward


# Flaten layer

class Flatten:
    def __init__(self, name='flat'):
        self.name = name
        self.trainable = None

    def __call__(self, inp):
        self.inp = inp

        return inp.flatten()

    def backward(self, dl_da):
        return np.reshape(dl_da, newshape=self.inp.shape).astype(np.float64)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))




# FC layer

class Dense:
    def __init__(self, neurons=1, activation='', name='Dense', trainable=True):
        self.neurons = neurons
        self.activation = activation.lower()
        self.w = None
        self.b = None
        self.name = name
        self.trainable = trainable

    def __call__(self, inputs):
        self.inputs = inputs
        if self.w is None:
            self.w = np.random.uniform(-1, 1, size=(len(inputs), self.neurons)) 
            self.b = np.random.uniform(-1, 1, self.neurons) 

        self.z = np.array(self.inputs @ self.w + self.b)

        if self.activation in activations.keys():
            self.a = activations[self.activation](self.z)
        else:
            self.a = self.z

        self.a = np.array(self.a)

        return self.a

    def backward(self, dl_da):
        
        if self.activation in activation_derivatives.keys():
            grad_z = dl_da * activation_derivatives[self.activation](self.z)
        else:
            grad_z = dl_da

        grad_w =  self.inputs.reshape(1, -1).T @ grad_z.reshape(1, -1)
        
        
        grad_b = grad_z.squeeze()


        grad_a = grad_z @ self.w.T

        return grad_a if not self.trainable else (grad_a, (grad_w, grad_b), (self.w, self.b))


