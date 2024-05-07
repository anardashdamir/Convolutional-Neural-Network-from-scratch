from layers import Conv, Flatten, Dense, MaxPool, ReLU
from optimizers import SGD, Adam
from model import Network
import numpy as np

import torchvision.datasets as datasets





mnist_trainset = datasets.MNIST(root=r'data', train=True, download=True, transform=None)

images = np.expand_dims(mnist_trainset.data.numpy(), axis=-1)
labels = np.expand_dims(mnist_trainset.train_labels.numpy(), axis=-1)

images = images[(mnist_trainset.train_labels == 0) | (mnist_trainset.train_labels == 1)]
labels = labels[(mnist_trainset.train_labels == 0) | (mnist_trainset.train_labels == 1)]



model = Network([
    Conv(n_filters=2, stride=1),
    ReLU(),
    MaxPool(filter_size=2),
    Flatten(),
    Dense(1, 'sigmoid', trainable=True)

])

model.optimizer = Adam(lr=0.01)

model.fit(images[:200]/255, labels[:200], epochs=100)
