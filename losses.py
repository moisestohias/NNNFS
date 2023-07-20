# looses.py
import numpy as np 
from layer import Layer as _Layer

# Functional Losses
def mse(y, p): return 0.5*np.power((y-p), 2).mean()
def mseP(y, p): return 2*(p-y)/np.prod(y.shape)
def ce(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return (- y * np.log(p) - (1 - y) * np.log(1 - p)).mean()
def ceP(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return  - (y / p) + (1 - y) / (1 - p)
def bce(y, p): return (-y * np.log(p) - (1 - y) * np.log(1 - p)).mean()
def bceP(y, p): return ((1 - y) / (1 - p) - y / p) / np.size(y)

def backward_crossentropy(top_grad, x, y):
    res = np.zeros(x.shape, dtype=x.dtype)
    res[np.arange(x.shape[0]), y] = - np.reciprocal(x[np.arange(x.shape[0]), y]) / x.shape[0]
    return res * top_grad

def softmax_crossentropy(x, y):
    s = softmax(x)
    return crossentropy(s, y), s

def backward_softmax_crossentropy(top_grad, inp_softmax, y):
    res = inp_softmax
    res[np.arange(res.shape[0]), y] -= 1
    return top_grad * res / inp_softmax.shape[0]
def crossentropy(x, y): return np.mean(-np.log(x[np.arange(x.shape[0]), y]))

# OOP ========================
class Layer(_Layer):
    def __init__(self, arg):
    super(Layer,Layer).__init__()
    def acc(self, y, p): return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

class MSE(Layer):
    def forward(self, y, p): return mse(y, p)
    def backward(self, y, p): return mseP(y,p)
class CrossEntropy(Layer):
    def forward(self, y, p): return ce(y, p)
    def backward(self, y, p): return ceP(y, p)
class BCrossEntropy(Layer):
    def forward(self, y, p): return bce(y, p)
    def backward(self, y, p): return bceP(y, p)




