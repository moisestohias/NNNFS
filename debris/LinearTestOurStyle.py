# outStyle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(41)

def mse(pred, y): return ((pred-y)**2).mean()
def mseP(pred, y): return (pred-y)/y.shape[0]*0.5  # F(G(x))'' -> G(x)' * F(G(x))': 2*(pred-y)

def affine_transform(Z, W, B=0): return Z.dot(W) + B # W: (In,Out) vs torch convention (In,Out)
def backward_affine_transform(TopGrad, Z, W):
    BGrad = TopGrad.sum(axis=0)
    WGrad = Z.T.dot(TopGrad)
    Zgrad = TopGrad.dot(W.T)
    return Zgrad, WGrad, BGrad

class Linear:
    def __init__(self, inF, outF): self.weights  = np.random.randn(inF, outF)
    def __call__(self, x): return self.forward(x)

    def forward(self, input):
        self.input = input
        return self.input.dot(self.weights)

    def backward(self, TopGrad):
        WGrad = self.input.T.dot(TopGrad)
        Zgrad = TopGrad.dot(self.weights.T)
        return Zgrad, WGrad

torch.manual_seed(41)
x = torch.rand(100,90)
y = torch.rand(100,40)
 
TL = nn.Linear(90,40, bias=False)
tpred = TL(x)
tloss = F.mse_loss(tpred, y)
tloss.backward()

ML = Linear(90,40)
ML.weights = TL.weight.detach().numpy().T
mpred = ML(x.numpy()) # [10,6] [6,4] 
mloss = mse(mpred, y.numpy())

# Backward
mseGrad = mseP(mpred, y.numpy())
mZgrad, mWGrad = ML.backward(mseGrad)
(TL.weight.grad.detach().numpy() - mWGrad.T).mean()