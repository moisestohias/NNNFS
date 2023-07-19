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
    def backward(self, ZPooled):
        KH, KW = self.K
        N, C, ZPH, ZPW, = ZPooled.shape #NCHW
        ZPNs, ZPCs, ZPHs, ZPWs = ZPooled.strides
        output_shape = (N,C,ZPH, ZPW, KH, KW) # intermediary segmented
        strides = (ZPNs,ZPCs,ZPHs,ZPWs,0,0)
        a = np.ascontiguousarray(as_strided(ZPooled, shape=output_shape, strides=strides))
        return a.transpose(0,1,2,4,3,5).reshape(N, C, ZPH*KH, ZPW*KW) # final output shape


np.random.seed(199)
K = (2,2)
KH, KW = K  #HW
n = 28
Z = np.arange(6*6*2).reshape(1,2,6,6)
# Z = np.random.randint(1,10,(1,1,6,6)).astype(np.float32)
# TZ = torch.as_tensor(Z)
A = newOrdering(inputArray=Z, verbose=True)

# Forward
# MP = MaxPool2d(Z.shape, K)
# ZPooled = MP(Z)
# TZPooled = F.max_pool2d(TZ, (2,2))
# print(TZPooled.shape)
# print(ZPooled.shape)

# Backward
# print(Z.squeeze())
# print(ZPooled.squeeze())
# print(MP.backward(ZPooled).squeeze())


