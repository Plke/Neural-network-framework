import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def softmax(z):
    a=np.max(z,axis=-1)
    a = np.expand_dims(a, axis=-1)
    z=z-a
    return np.exp(z) / (np.sum(np.exp(z), axis=-1, keepdims=True))

def relu(z):
    return z * (z > 0)

