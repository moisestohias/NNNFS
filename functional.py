# functional.py

import numpy as np
from math import ceil, floor
as_strided = np.lib.stride_tricks.as_strided

# LeakyReluAlpha = 0.02
# gain = dict(Tanh=5/3, ReLU=np.sqrt(2), LeakyReLU=1/np.sqrt(1+LeakyReluAlpha**2), SELU=3/4)

def _affTrans(Z, W, B=0): return Z.dot(W) + B # W(inF,outF)
def _affTransP(TopGrad, Z, W):
    BGrad = TopGrad.sum(axis=0)
    WGrad = Z.T.dot(TopGrad)
    Zgrad = TopGrad.dot(W.T)
    return Zgrad, WGrad, BGrad

# Note: torch convention w(outF,inF), requiring forward: Z.dot(W.T)+B, and backward: TopGrad.T.dot(Z) & TopGrad.dot(W) instead.
def affTrans(Z, W, B=0): return Z.dot(W.T) + B # W: (outF,inF)
def affTransP(TopGrad, Z, W):
    BGrad = TopGrad.sum(axis=0)
    WGrad = TopGrad.T.dot(Z)
    Zgrad = TopGrad.dot(W)
    return Zgrad, WGrad, BGrad


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

def calculateConvOutShape(inShape, KS, S=(1,1), P=(0,0), D=(1,1)):
    Hin, Win = inShape
    Hout = floor(((Hin+2*P[0]-D[0]*(KS[0]-1)-1)/S[0]) +1)
    Wout = floor(((Win+2*P[1]-D[1]*(KS[1]-1)-1)/S[1]) +1)
    outShape = Hout, Wout
    return outShape

def calculateTranspConvOutShape(inShape, KS, S=(1,1), P=(0,0), OP=(0,0), D=(1,1)):
    """
    KS: (KH,KW) kernel_size
    S: (SH,SW) stride
    P: (PH,PW) padding
    OP: (PH,PW) output_padding
    D: (DH,DW) dilation
    """
    Hin, Win = inShape
    Hout =(Hin -1)*S[0]-2*P[0]+D[0]*(KS[0]-1)+OP[0]+1
    Wout =(Win -1)*S[1]-2*P[1]+D[1]*(KS[1]-1)+OP[1]+1
    return Hout, Wout

def _corr2d_Old(Z, W):
    # Add dilation & stride support
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

def _corr2d(Z, W, S=(1,1), P=(0,0), D=(1,1)):
    """ Convolution with stride and padding support
    Z: (N,C_in,H,W)
    W: (C_out,C_in,KH,KW)
    S: (SH,SW) stride
    P: (PH,PW) padding
    D: (DH,DW) dilation
    The reason we permute the dimensions is to make HWC of Z align with KKI of W as the last dim.
    """
    Z = Z.transpose(0,2,3,1) # NCHW -> NHWC
    W = W.transpose(2,3,1,0) # OIKK -> KKIO
    N,ZH,ZW,C_in = Z.shape
    KH,KW,_,C_out = W.shape
    Ns, ZHs, ZWs, Cs = Z.strides
    outH, outW = calculateConvOutShape((ZH,ZW), (KH,KW), S, P, D)
    outShape = (N, outH, outW, KH, KW, C_in)
    inner_dim = KH * KW * C_in # Size of kernel flattened
    A = as_strided(Z, shape = outShape, strides = (Ns, ZHs*S[0], ZWs*S[0], ZHs*D[0], ZWs*D[1], Cs)).reshape(-1,inner_dim)
    out = A @ W.reshape(-1, C_out)
    return out.reshape(N,outH,outW,C_out).transpose(0,3,1,2) # NHWC -> NCHW

# Pooling
def pool1d(Z, K:int=2, Pad=False):
    N, C, W = Z.shape #NCHW
    Ns, Cs, Ws = Z.strides
    EdgeW = W%K # How many pixels left on the edge
    if Pad and EdgeW!=0: # If there are pixelx left and Pad=True, we pad
        PadW = K-EdgeW
        PadLeft, PadRight = ceil(PadW/2), floor(PadW/2)
        Z = np.pad(Z, ((0,0),(0,0), (PadLeft, PadRight)))
        N, C, W = Z.shape #NCHW
        Ns, Cs, Ws = Z.strides
    return as_strided(Z, shape=(N,C, W//K, K), strides=(Ns, Cs, Ws*K , Ws) )

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

def unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
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
    return Z



def maxpool2d_backward(d_pool, padded, pool_size=(2,2)): return unpool2d(d_pool, padded, pool_size)

def unpool1d(Z, K):
    N,C,W= Z.shape
    return np.ascontiguousarray(as_strided(Z, (*Z.shape,K), (*Z.strides,0)).reshape(N, C, W*K))
def unpool2d(Z, K):
    KH, KW = K
    N, C, ZPH, ZPW = Z.shape #NCHW
    a = np.ascontiguousarray(as_strided(Z, (*Z.shape,KH,KW), (*Z.strides,0,0)))
    return a.transpose(0,1,2,4,3,5).reshape(N, C, ZPH*KH, ZPW*KW)

def avgpool1d(Z, K:int=2): return pool1d(Z, K).mean(axis=-1)
def maxpool1d(Z, K:int=2): return pool1d(Z, K).max(axis=-1)
def avgpool2d(Z, K:tuple=(2,2)): return pool2d(Z, K).mean(axis=(-2,-1)) # for average pool, this might work


def lstm_cell(Z, H, C, W_hh, W_ih, B):
    i,f,g,o = np.split(Z@W_ih + H@W_hh + B[None,:], 4, axis=1) # Input, Forget,g (tanh-Activation) , Output
    i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f*c + i*g
    h_out = o * np.tanh(c_out)
    cache = i,f,o,g, c_out, C,x , h_out, Wx, Wh
    return h_out, c_out

def lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        H[t,:,:] = h # Batch Comes second for contiguous memory :,:
    return H, c

def normalize(Z): return (Z-np.mean(Z))/np.std(Z) # standardize really

# Losses
def cross_entropy(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return - y * np.log(p) - (1 - y) * np.log(1 - p)
def cross_entropy_prim(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return  - (y / p) + (1 - y) / (1 - p)
def binary_cross_entropy(y, p): return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))
def binary_cross_entropy_prime(y, p): return ((1 - y) / (1 - p) - y / p) / np.size(y)

def mse(y, p): return 0.5*np.power((y-p), 2).mean()
def mse_prime(y, p): return 2*(p-y)/np.prod(y.shape)
# Note: torch standard is:
# def mse(p, y): return 0.5*np.power((p-y), 2).mean()
# def mse_prime(p, y): return (y-p).mean()


# Activations
def sigmoid(x): return np.reciprocal((1.0+np.exp(-x)))
def sigmoid_prime(x): s = np.reciprocal((1.0+np.exp(-x))); return s * (1 - s) # σ(x)*(1-σ(x))
def relu(x): return np.where(x>= 0, x, 0)
def relu_prime(x): return np.where(x>= 0, 1, 0)
def leaky_relu(x, alpha=0.01): return np.where(x>= 0, x, alpha*x)
def leaky_relu_prime(x, alpha=0.01): return np.where(x>= 0, 1, alpha)
def elu(x, alpha=0.01): return np.where(x>= 0, x, alpha*(np.exp(x)-1))
def elu_prime(x, alpha=0.01): return np.where(x>= 0, 1, alpha*np.exp(x))
def swish(x): return x * np.reciprocal((1.0+np.exp(-x))) # x*σ(x) σ(x)+σ'(x)x : σ(x)+σ(x)*(1-σ(x))*x
def swish_prime(x): s = np.reciprocal((1.0+np.exp(-x))); return s+s*(1-s)*x #σ(x)+σ(x)*(1-σ(x))*x
silu, silu_prime = swish, swish_prime # The SiLU function is also known as the swish function.
def tanh(x): return np.tanh(x) # or 2.0*(σ((2.0 * x)))-1.0
def tanh_prime(x): return 1 - np.tanh(x) ** 2
def gelu(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*np.power(x,3)))) # sqrt(2/pi)=0.7978845608
def gelu_prime(x): return NotImplemented#Yet
def quick_gelu(x): return x*sigmoid(x*1.702) # faster version but inacurate
def quick_gelu_prime(x): return 1.702*sigmoid_prime(x*1.702)
def hardswish(x): return x*relu(x+3.0)/6.0
def hardswish_prime(x): return 1.0/6.0 *relu(x+3)*(x+1.0)
def softplus(x, limit=20.0, beta=1.0): return (1.0/beta) * np.log(1 + np.exp(x*beta))
def softplus_prime(limit=20, beta=1.0): _s = np.exp(x*beta) ; return (beta*_s)/(1+_s)
def relu6(x): return relu(x)-relu(x-6)
def relu6_prime(x): return relu_prime(x)-relu_prime(x-6)


"""
# Test Conv
np.random.seed(42)
Z = np.random.randn(1,1,8,8).astype(np.float32)
W = np.random.randn(1,1,3,3).astype(np.float32)
TZ, TW = torch.as_tensor(Z), torch.as_tensor(W)
TZ.requires_grad_(), TW.requires_grad_()

# Forkward
Tout = F.conv2d(TZ, TW)
out = corr2d(Z, W)
print(out.shape==Tout.shape, mag(out-Tout.detach().numpy()).round(4))

# Backward
TopGrad = np.ones_like(out)
Tout.backward(torch.as_tensor(TopGrad))
WGrad, ZGrad = corr2d_backward(Z, W, TopGrad)
print(TZ.grad.shape == ZGrad.shape, mag(TZ.grad.numpy() - ZGrad).round(4))

"""


def crossentropy(x, y): return np.mean(-np.log(x[np.arange(x.shape[0]), y]))
def softmax(x):
    temp = np.exp(x - x.max(axis=1, keepdims=True))
    res = temp / temp.sum(axis=1, keepdims=True)
    return res

def backward_softmax(top_grad, inp_softmax):
    left = inp_softmax[:, :, np.newaxis]
    right = inp_softmax[:, np.newaxis, :]
    sub = left * np.eye(inp_softmax.shape[1])
    mul = np.matmul(left, right)
    res = np.matmul((sub - mul), top_grad[:, :, np.newaxis]).squeeze()
    return res

def backward_crossentropy(top_grad, x, y):
    res = np.zeros(x.shape, dtype=x.dtype)
    res[np.arange(x.shape[0]), y] = - np.reciprocal(x[np.arange(x.shape[0]), y]) / x.shape[0]
    return res * top_grad

def softmax_crossentropy(x, y):
    s = softmax(x)
    return crossentropy(s, y), s

def backward_softmax_crossentropy(top_grad, inp_softmax, y):
    res = inp_softmax
    res[np.arange(res.shape[0]), y] -= 1
    return top_grad * res / inp_softmax.shape[0]


