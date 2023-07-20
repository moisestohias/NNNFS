# looses.py
import numpy as np 

# Functional Losses
def mse(y, p): return 0.5*np.power((y-p), 2).mean()
def mseP(y, p): return 2*(p-y)/np.prod(y.shape)
def ce(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return - y * np.log(p) - (1 - y) * np.log(1 - p)
def ce_prim(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return  - (y / p) + (1 - y) / (1 - p)
def bce(y, p): return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))
def bceP(y, p): return ((1 - y) / (1 - p) - y / p) / np.size(y)
def bce(y, p): return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))
def bceP(y, p): return ((1 - y) / (1 - p) - y / p) / np.size(y)

# OOP Losses
class MSE(Layer):
    def forward(self, y, p): return mse(y, p)
    def backward(self, y, p): return mseP(y,p)

class CrossEntropy(Layer):
    # This class only calculate the loss for a single input, to vectorize, you need to take the mean in the loss, deviding by y.size gradient.
    def forward(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15) # Avoid division by zero
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def backward(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return  - (y / p) + (1 - y) / (1 - p)

    def acc(self, y, p): return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))



