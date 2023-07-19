# I am writing a tiny DL library in pure Numpy. I want you to write 3 different implementation of the backward version of maxpool2d, while focusing on effeciency
import numpy as np
from math import ceil, floor
as_strided = np.lib.stride_tricks.as_strided

# Functional
def mse(pred, y): return ((pred-y)**2).mean()
def mseP(pred, y): return 2*(pred-y)/np.prod(y.shape)  # F(G(x))'' -> G(x)' * F(G(x))': 2*(pred-y)

def _pad(Z: np.ndarray, K: np.ndarray, mode: str="valid") -> np.ndarray:
    """ Check arguments and pad for conv/corr """
    if mode not in ["full", "same", "valid"]: raise ValueError("mode must be one of ['full', 'same', 'valid']")
    if Z.ndim != K.ndim: raise ValueError("Z and K must have the same number of dimensions")
    if Z.size == 0 or K.size == 0: raise ValueError(f"Zero-size arrays not supported in convolutions.")
    ZN,ZC,ZH,ZW = Z.shape
    OutCh,KC,KH,KW = K.shape
    if ZC!=KC: raise ValueError(f"Kernel must have the same number channels as Input, got Z.shape:{Z.shape}, W.shape {K.shape}")
    if mode == 'valid' : padding = ((0,0),(0,0), (0,0), (0,0))
    elif mode == 'same':
        # OH = ZH-KH+1 -> ZH=OH+KH-1
        PadTop, PadBottom = floor((KH-1)/2), ceil((KH-1)/2)
        PadLeft, PadRigh = floor((KW-1)/2), ceil((KW-1)/2)
        padding = ((0,0),(0,0), (PadTop, PadBottom),(PadLeft, PadRigh))
    elif mode == 'full':
        PadTop, PadBottom = KH-1, KH-1 # full-convolution aligns kernel edge with the firs pixel of input, thus K-1
        PadLeft, PadRigh = KW-1, KW-1
        padding = ((0,0),(0,0), (PadTop, PadBottom),(PadLeft, PadRigh))
    if np.array(padding).any(): Z = np.pad(Z, padding)
    return Z, K

def _corr2d(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    Z = Z.transpose(0,2,3,1) # NCHW -> NHWC
    W = W.transpose(2,3,1,0) # OIKK -> KKIO

    N,ZH,ZW,C_in = Z.shape
    KH,KW,_,C_out = W.shape
    Ns, ZHs, ZWs, Cs = Z.strides

    inner_dim = KH * KW * C_in # Size of kernel flattened
    A = as_strided(Z, shape = (N, ZH-KH+1, ZW-KW+1, KH, KW, C_in), strides = (Ns, ZHs, ZWs, ZHs, ZWs, Cs)).reshape(-1,inner_dim)
    out = A @ W.reshape(-1, C_out)
    return out.reshape(N,ZH-KH+1,ZW-KW+1,C_out).transpose(0,3,1,2) # NHWC -> NCHW

def conv2d(Z, W, mode:str="valid"): return _corr2d(*_pad(Z, np.flip(W), mode))
def corr2d(Z, W, mode:str="valid"): return _corr2d(*_pad(Z, W, mode))
def corr2d_backward(Z, W, TopGrad,  mode:str="valid"):
    WGrad = corr2d(Z.transpose(1,0,2,3), TopGrad.transpose(1,0,2,3)).transpose(1,0,2,3)
    ZGrad = np.flip(np.rot90(corr2d(TopGrad, W.transpose(1,0,2,3), "full"))).transpose(1,0,2,3)
    return WGrad , ZGrad

def pool2d(Z, K:tuple=(2,2), MustPad=False):
    KH, KW = K  # Kernel Height & Width
    N, C, ZH, ZW = Z.shape # Input: NCHW Batch, Channels, Height, Width
    Ns, Cs, Hs, Ws = Z.strides
    EdgeH, EdgeW = ZH%KH, ZW%KW # How many pixels left on the edge
    if MustPad and (EdgeH!=0 or EdgeW!=0): # If there are pixelx left and Pad=True, we pad
        PadH, PadW = KH-EdgeH, KW-EdgeW
        PadTop, PadBottom = ceil(PadH/2), floor(PadH/2)
        PadLeft, PadRight = ceil(PadW/2), floor(PadW/2)
        Z = np.pad(Z, ((0,0),(0,0), (PadTop, PadBottom), (PadLeft, PadRight)))
        N, C, ZH, ZW = Z.shape #NCHW
        Ns, Cs, Hs, Ws = Z.strides
    Zstrided = as_strided(Z, shape=(N,C,ZH//KH, ZW//KW, KH, KW), strides=(Ns, Cs, Hs*KH, Ws*KW,Hs, Ws))
    return Zstrided.reshape(N,C,ZH//KH, ZW//KW, KH*KW) # reshape to flatten pooling windows to be 1D-vector

def maxpool2d(Z, K:tuple=(2,2)):
    ZP = pool2d(Z, K)
    MxP = np.max(ZP, axis=(-1))
    Inx = np.argmax(ZP, axis=-1)
    return MxP, Inx


def affTrans(Z, W, B=0): return Z.dot(W.T) + B # W: (outF,inF)
def affTransP(TopGrad, Z, W):
    BGrad = TopGrad.sum(axis=0)
    WGrad = TopGrad.T.dot(Z)
    Zgrad = TopGrad.dot(W)
    return Zgrad, WGrad, BGrad

class Layer:
    """ All layers should Only acccept batch of inputs: (N,C,H,W)"""
    def __call__(self, x): return self.forward(x)
    def forward(self, input): raise NotImplementedError
    def backward(self, TopGrad): raise NotImplementedError

class MSELoss(Layer):
    def forward(self, pred, y): return mse(pred, y)
    def backward(self, pred, y): return mseP(pred, y)

class Linear(Layer):
    def __init__(self, inF, outF, bias=True):
        self.layers_name = self.__class__.__name__
        self.trainable = True
        lim = 1 / np.sqrt(inF)
        self.weight  = np.random.uniform(-lim, lim, (outF, inF))
        self.bias = np.random.randn(outF) * 0.1 if bias else None
        self.params = [self.weight, self.bias]
        # self.output_shape = inF, outF

    def forward(self, input):
        self.input = input
        return affTrans(self.input, self.weight, self.bias) # (MBS,inF)x(outF,inF).T -> (MBS,outF)

    def backward(self, TopGrad):
        self.Zgrad, self.WGrad, self.BGrad = affTransP(TopGrad, self.input, self.weight)
        self.weight -= self.WGrad*LR
        self.BGrad  -= self.WGrad*LR
        return self.Zgrad


class Conv2d(Layer):
    def __init__(self, inCh, outCh, KS, stride=1, padding=0, dilation=1, groups=1, bias=True, inShape=None):
        if isinstance(KS, int): KS = (KS, KS)
        if isinstance(stride, int): stride = (stride, stride)
        if isinstance(padding, int): padding = (padding, padding)
        if isinstance(dilation, int): dilation = (dilation, dilation)
        if inShape: 
            Hin, Win = inShape
            Hout = floor((Hin+2*padding[0]*dilation[0]*(KS[0]-1)-1+1)/stride[0])
            Wout = floor((Win+2*padding[0]*dilation[1]*(KS[1]-1)-1+1)/stride[1])
            self.outShape = outCh, Hout, Wout

        self.layers_name = self.__class__.__name__
        self.trainable = True
        self.weight = np.random.randn(outCh, inCh, *KS) # (outCh,inCh,H,W)
        self.bias = np.random.randn(outCh) # Each filter has bias, not each conv window
        self.params = [self.weight, self.bias]

    def forward(self, x):
        self.input = x
        return corr2d(x, self.weight) + self.bias[np.newaxis, :, np.newaxis, np.newaxis]

    def backward(self, TopGrad):
        kernels_gradient, input_gradient = corr2d_backward(self.input, self.weight, TopGrad)
        self.grads = (kernels_gradient, TopGrad.sum(axis=(0, 2, 3)))
        return input_gradient


MBS = 20
x, y = np.random.randn(10, 784), np.arange(10)
