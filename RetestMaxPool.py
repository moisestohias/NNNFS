import numpy as np
as_strided = np.lib.stride_tricks.as_strided
from math import ceil, floor

def pool2d(Z, K:tuple=(2,2)):
    """ performs the windowing, and padding if needed
    !Note: most implementations including Pytorch don't pad,
    if there are pixels left just drop them. We may need to reconsider.
    """
    KH, KW = K  # Kernel Height & Width
    N, C, ZH, ZW = Z.shape # Input: NCHW Batch, Channels, Height, Width
    Ns, Cs, Hs, Ws = Z.strides
    PadBottom, PadRight = ZH%KH, ZW%KW # How many pixels left on the edge
    Padd = ((0,0),(0,0), (0, PadBottom), (0, PadRight))
    Zstrided = as_strided(Z, shape=(N,C,ZH//KH, ZW//KW, KH, KW), strides=(Ns, Cs, Hs*KH, Ws*KW,Hs, Ws))
    return Zstrided.reshape(N,C,ZH//KH, ZW//KW, KH*KW), Padd


def maxpool2d(Z, K:tuple=(2,2)):
    ZP, Padd = pool2d(Z, K)
    MxP = np.max(ZP, axis=(-1))
    Inx = np.argmax(ZP, axis=-1)
    return MxP, Inx, Padd

def unmaxpool2d(ZP, Indx, Padd, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))

    for n in range(ZN):
        for c in range(ZC):
            for h in range(ZH):
                for w in range(ZW):
                    ind = Indx[n, c, h, w]  # index of max value in pooling window
                    row, col = np.unravel_index(ind, (KH, KW))
                    Z[n, c, h*KH+row, w*KW+col] = ZP[n, c, h, w]

    Z = np.pad(Z, Padd)
    return Z


np.random.seed(42)
K = (2,2)
Z = np.random.randint(1,10,(1,1,5,5)).astype(np.float32)
ZP, Indx, Padd = maxpool2d(Z, K) # Forward
ZUnp = unmaxpool2d(ZP, Indx, Padd, K) # Backward
print(Z.squeeze())
print(ZUnp.squeeze())
