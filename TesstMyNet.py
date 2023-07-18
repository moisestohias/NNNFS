# tesst.py

import numpy as np
from math import ceil, floor
as_strided = np.lib.stride_tricks.as_strided
from functional import corr2d, conv2d, corr2d_backward #, corr2d_backward_conv

def corr2d_backward_conv(Z, W, TopGrad,  mode:str="valid"):
    WGrad = corr2d(Z.transpose(1,0,2,3), TopGrad.transpose(1,0,2,3)).transpose(1,0,2,3)
    ZGrad = conv2d(TopGrad.transpose(0,2,3,1), W.transpose(0,2,3,1), "full").transpose(0,3,1,2)
    return WGrad , ZGrad
def corr2d_backward_conv(Z, W, TopGrad,  mode:str="valid"):
    WGrad = corr2d(Z.transpose(1,0,2,3), TopGrad.transpose(1,0,2,3)).transpose(1,0,2,3)
    ZGrad = conv2d(TopGrad, W.transpose(0,1,3,2), "full")
    return WGrad , ZGrad


Z = np.random.rand(1,1,9,9)
W = np.random.rand(1,1,3,3)
A = corr2d(Z, W)
print(A.shape)
TGrad = np.ones_like(A)
WGrad, ZGrad = corr2d_backward(Z, W, TGrad)
WGradC, ZGradC = corr2d_backward_conv(Z, W, TGrad)
print(ZGrad.shape)
# print(ZGradC.shape)
# print(np.linalg.norm(ZGrad-ZGradC))
