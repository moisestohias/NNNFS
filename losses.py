# looses.py
import numpy as np 
from layer import Layer as _Layer

# Functional Losses
# TODO: protect agains overflow: clip and add epsilon: p = np.clip(p, 1e-15, 1 - 1e-15); log(x + eps)
def mse(y, p): return 0.5*np.power((y-p), 2).mean()
def mseP(y, p): return 2*(p-y)/np.prod(y.shape)
def ce(y, p): return (-np.log(p)).mean()
def ce(y, p): ...
def bce(y, p): return (-y * np.log(p)-(1-y)*np.log(1-p)).mean()
def bceP(y, p): return ((1 - y) / (1 - p) - y / p) / y.shape[0]

def backward_crossentropy(top_grad, x, y):
    res = np.zeros(x.shape, dtype=x.dtype)
    res[np.arange(x.shape[0]), y] = - np.reciprocal(x[np.arange(x.shape[0]), y]) / x.shape[0]
    return res * top_grad


# we must clip
def softmax(x): s = np.exp(x-x.max(axis=-1, keepdims=True)); return s / s.sum(axis=-1, keepdims=True)
def softmaxP(s): s = s.reshape(-1, 1); return np.diagflat(s) - np.dot(s, s.T)
def logSoftmax(x): s = np.exp(x-x.max()); return s - np.log(s.sum(axis=-1, keepdims=True))
def logSoftmaxP(x): ...
def softmaxCE(y, pred): return -(np.sum(y * np.log(pred + 1e-12), axis=-1)).mean()
def softmaxCEP(y, s): return (s-y)/y.shape[0]

def softmaxCE(x, y): 
    x = x-x.max(axis=-1, keepdims=True) # keepdims in the sum, can be set to true or false both will work
    return (np.log(np.exp(x).sum(axis=-1))-x[np.arange(y.shape[0]), y]).mean()

# def softmaxCE(x, y): s = softmax(x); return crossentropy(s, y), s
def softmaxCE(x, y): softmax(x) - y

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




