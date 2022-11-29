import numpy as np


def cross_entropy(a, y):
    a = np.where(a > 1 - 1e-4, 1 - 1e-4, a)
    return -(y * np.log(a + 1e-5))


def mse(a, y):
    return (np.square(y - a))


