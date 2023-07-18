import numpy as np
from math import ceil, floor
as_strided = np.lib.stride_tricks.as_strided
import torch.nn.functional as F
import torch


class Layer:
    def __init__(self): pass
    def __call__(self, x): return self.forward(x)
    def forward(self, input): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError

class MaxPool2d(Layer):
    """ Only batch input is supported NCHW"""
    def __init__(self, ZShape, K:tuple=(2,2)):
        self.MustPad = False
        self.N, self.C, ZH, ZW = ZShape #NCHW
        self.K = K  #HW
        EdgeH, EdgeW = ZH%KH, ZW%KW # How many pixels left on the edge
        if (EdgeH!=0 or EdgeW!=0): # If there are pixelx left we need to pad
            self.MustPad = True
            PadH, PadW = KH-EdgeH,KW-EdgeW
            PadTop, PadBottom = ceil(PadH/2), floor(PadH/2)
            PadLeft, PadRight = ceil(PadW/2), floor(PadW/2)
            self.Padding = ((0,0),(0,0), (PadTop, PadBottom), (PadLeft, PadRight))
    def forward(self,Z):
        Ns, Cs, Hs, Ws = Z.strides
        N, C, ZH, ZW = Z.shape #NCHW
        if self.MustPad:
            Z = np.pad(Z, self.Padding)
            N, C, ZH, ZW = Z.shape #NCHW
            Ns, Cs, Hs, Ws = Z.strides
        return as_strided(Z, shape=(N,C,ZH//KH, ZW//KW, KH, KW), strides=(Ns, Cs, Hs*KW, Ws*KH,Hs, Ws)).max(axis=(-2,-1))
    def backward(self, ZP):
        KH, KW = self.K
        N, C, ZPH, ZPW, = ZP.shape #NCHW
        ZPNs, ZPCs, ZPHs, ZPWs = ZP.strides
        a = as_strided(ZP, shape=(N,C,ZPH, ZPW, KH, KW), strides=(ZPNs,ZPCs,ZPHs,ZPWs,0,0))
        return a.transpose(0,1,2,4,3,5).reshape(N, C, ZPH*KH, ZPW*KW) # final output shape


# np.random.seed(199)
# K = (2,2)
# KH, KW = K  #HW
# n = 28
# # z = np.arange(6*6).reshape(6,6)
# # Z = np.stack((z,z.T,z),0).reshape(1,3,6,6)
# # print(Z.shape)
# # raise SystemExit
# Z = np.random.randint(1,10,(2,1,6,6)).astype(np.float32)
# TZ = torch.as_tensor(Z)

# # Forward
# MP = MaxPool2d(Z.shape, K)
# ZPooled = MP(Z)
# # TZPooled = F.max_pool2d(TZ, (2,2)).numpy()
# # print(TZPooled.shape)
# # print(ZPooled.shape)
# # print(np.linalg.norm(TZPooled-ZPooled).round(8))

# # Backward
# print(Z.squeeze())
# print(ZPooled.squeeze())
# print(MP.backward(ZPooled).squeeze())

np.random.seed(42)
K = (2,2)
Z = np.random.randint(1,10,(1,1,6,6)).astype(np.float32)

ZP, Indx = maxpool2d(Z, K) # Forward
ZPooled = unmaxpool2d(ZP, K) # Backward

# TZPooled = F.max_pool2d(TZ, (2,2))
# print(TZPooled.shape)
# print(ZPooled.shape)

# TZ = torch.as_tensor(Z)
# Backward
print(ZZ.squeeze())
print(ZPooled.squeeze())
print(MP.backward(ZPooled).squeeze())
