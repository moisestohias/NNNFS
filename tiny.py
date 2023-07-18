# tiny.py

import numpy as np
# import torch
"""
Class Network/Model: layers
Class Layer: (Linear, Convolutional, Dropout, BatchNorm)
Layer -> ActLayer -> Layer -> ActLayer -> Layer
"""
# Functional
def sig(x): return np.reciprocal(1.0+np.exp(-x))
def sigP(x): s = sig(x); return s(1.0-s)
def relu(x): return np.where(x>=0, x, 0)
def reluP(x): return np.where(x>=0, 1, 0)
def leakyRelu(x, alpha=0.02): return np.where(x>=0, x, x*alpha)
def leakyReluP(x): return np.where(x>=0, 1, alpha)
def tanh(x): return np.tanh(x)
def tanhP(x): return 1 - np.tanh(x) ** 2

class Layer:
  def __repr__(self): return self.__class__.__name__
  def __call__(self,x ): return self.forward(x)
  def forward(self, x): raise NotImplementedError
  def backward(self, x): raise NotImplementedError

class Sigmoid(Layer): 
  def forward(self,x): return sig(x)
  def backward(self,topGrad): return sigP(topGrad)
class Tanh(Layer): 
  def forward(self,x): return tanh(x)
  def backward(self,topGrad): return tanhP(topGrad)
class ReLU(Layer): 
  def forward(self,x): return relu(x)
  def backward(self,topGrad): return reluP(topGrad)
class LeakyReLU(Layer): 
  def forward(self,x): return leakyRelu(x)
  def backward(self,topGrad): return leakyReluP(topGrad)

class Linear(Layer):
  def __init__(self,inF, outF): # a = z@w+b -> z[MBS, inF] w[inF, outF] -> [MBS, outF]
    self.bias = np.random.randn(outF)
    self.weight = np.random.randn(inF, outF)
    self.params = self.weight, self.bias
  def forward(self, x): self.x = x; return x.dot(self.weight)+self.bias
  def backward(self, topGrad): # a = z@w+b -> dL/dz= dL/dz @ w.T
    zGrad = topGrad.dot(self.weight.T) # dL/dz= dL/dz @ w.T
    self.wGrad =  self.x.T.dot(self.weight)# dL/dw= z.T @ w
    self.bGrad =  zGrad.sum(0)# dL/dw= z.T @ w
    return zGrad

class Linear(Layer):
  def __init__(self,inF, outF): # a = z@w+b -> z[MBS, inF] w[inF, outF] -> [MBS, outF]
    self.bias = np.random.randn(outF)
    self.weight = np.random.randn(inF, outF)
  def forward(self, x): self.x = x; return x.dot(self.weight)+self.bias
  def backward(self, topGrad): # a = z@w+b -> dL/dz= dL/dz @ w.T
    zGrad = topGrad.dot(self.weight.T) # dL/dz= dL/dz @ w.T
    self.wGrad =  self.x.T.dot(self.weight)# dL/dw= z.T @ w
    self.bGrad =  zGrad.sum(0)# dL/dw= z.T @ w
    self.weight -= self.wGrad*LR
    self.bias -= self.wBrad*LR
    return zGrad


MB = np.random.randn(1, 784)
TL = nn.Linear(784, 10).double()
ML = Linear(784, 10)
ML.weight = TL.weight.detach().numpy().T
ML.bias = TL.bias.detach().numpy()
tpred = TL(torch.as_tensor(MB)).detach().numpy()
mpred = ML(MB)
print(mpred.shape, tpred.shape)
print((mpred-tpred).sum())

class Net:
  def __call__(self,x ): return self.forward(x)
  def __init__(self):
    self.L1 = Linear(784, 100)
    self.L2 = Linear(100, 10)
    self.Act = Sigmoid()
    self.layers = [self.L1, self.Act, self.L2, self.Act]
  def forward(self, x):  
    for l in self.layers: x = l(x)
    return x
  def backward(self, topGrad): 
    for l in reversed(self.layers): topGrad = l(topGrad)
net = Net()
net(MB).shape
topGrad = np.ones(10)
net.backward()