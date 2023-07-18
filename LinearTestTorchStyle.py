# Torch style
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def affTrans(Z, W, B=0): return Z.dot(W.T) + B # W: (outF,inF)
def affTransP(TopGrad, Z, W):
    BGrad = TopGrad.sum(axis=0)
    WGrad = TopGrad.T.dot(Z)
    Zgrad = TopGrad.dot(W)
    return Zgrad, WGrad, BGrad

torch.manual_seed(41)
def mse(pred, y): return ((pred-y)**2).mean()
def mseP(pred, y): return 2*(pred-y)/(np.prod(y.shape))  # F(G(x))'' -> G(x)' * F(G(x))': 2*(pred-y)

class Linear:
    def __init__(self, inF, outF):
        self.weight  = np.random.randn(outF, inF)
        self.bias  = np.random.randn(outF)
    def __call__(self, x): return self.forward(x)

    def forward(self, input):
        self.input = input
        return affTrans(self.input, self.weight, self.bias) # [N,inF][outF,inF].T -> [N,inF][inF,outF] -> [N,outF]
        return self.input.dot(self.weight.T) + self.bias # [N,inF][outF,inF].T -> [N,inF][inF,outF] -> [N,outF]

    def backward(self, TopGrad):
        # self.BGrad = TopGrad.sum(axis=0)
        # self.WGrad = TopGrad.T.dot(self.input) # [N,outF].Tx[N,inF]  -> [outF, inF]
        # self.Zgrad = TopGrad.dot(self.weight) # [N,outF]x[outF, inF] -> [N, inF]
        self.Zgrad, self.WGrad, self.BGrad = affTransP(TopGrad, self.input, self.weight)
        return self.Zgrad

MBS = 20
inF, outF = 100, 10
torch.manual_seed(2)
x = torch.rand(MBS,inF, requires_grad=False)
y = torch.rand(MBS,outF, requires_grad=False)

TL = nn.Linear(inF,outF)
tpred = TL(x)
tloss = F.mse_loss(tpred, y)
tloss.backward()

ML = Linear(inF,outF)
ML.weight = TL.weight.detach().numpy()
ML.bias = TL.bias.detach().numpy()
mpred = ML(x.numpy()) # [MBS,inF] [inF,outF]
mloss = mse(mpred, y.numpy())

# print(mpred, tpred.detach().numpy(), sep="\n")
# print((mpred-tpred.detach().numpy()).sum())

# Backward
mseGrad = mseP(mpred, y.numpy())
mZgrad = ML.backward(mseGrad)
# print((TL.weight.grad.detach().numpy()/mWGrad).mean())
print((TL.weight.grad.detach().numpy()/ML.WGrad).mean())
