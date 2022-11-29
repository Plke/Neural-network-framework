import numpy as np


def Adagrad(eta, mom_w, grad_w, w, grad_b=None, mom_b=None, b=None, alpha=0.9):
    h_w = np.zeros_like(grad_w) + 1e-8

    h_b = np.zeros_like(grad_b) + 1e-8
    h_w += grad_w * grad_w
    w -= eta / np.sqrt(h_w) * grad_w

    h_b += grad_b * grad_b
    b -= eta / np.sqrt(h_b) * grad_b

    return w, b, mom_w, mom_b


def SGD(eta, mom_w, grad_w, w, grad_b=None, mom_b=None, b=None, alpha=0.9):
    w = w - eta * grad_w

    b = b - eta * grad_b

    return w, b, mom_w, mom_b


def Momentum(eta, mom_w, grad_w, w, grad_b=None, mom_b=None, b=None, alpha=0.9):
    global paramsv
    mom_w = alpha * mom_w - eta * grad_w
    w = w + mom_w

    mom_b = alpha * mom_b - eta * grad_b
    b = b + mom_b

    return w, b, mom_w, mom_b
