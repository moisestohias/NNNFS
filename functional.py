# functional.py
""" Should we keep all functional stuff here includding act/loss ...???"""
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

# Note: torch convention w(outF,inF), requiring forward: Z.dot(W.T)+B & backward: TopGrad.T.dot(Z) & TopGrad.dot(W).
# Maybe I'll ditch torch convention for the linear layer, doesn't make any sense 😕
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
        # TODO: change this to take into acound stride/dilation
        PadTop, PadBottom = floor((KH-1)/2), ceil((KH-1)/2)
        PadLeft, PadRigh = floor((KW-1)/2), ceil((KW-1)/2)
        padding = ((0,0),(0,0), (PadTop, PadBottom),(PadLeft, PadRigh))
    elif mode == 'full':
        PadTop, PadBottom = KH-1, KH-1 # full-convolution aligns kernel edge with the firs pixel of input, thus K-1
        PadLeft, PadRigh = KW-1, KW-1
        padding = ((0,0),(0,0), (PadTop, PadBottom),(PadLeft, PadRigh))
    if np.array(padding).any(): Z = np.pad(Z, padding)
    return Z, K

def calculateMaxpool2dOutShape(*a, **kw): return calculateConvOutShape(*a, **kw) # Correct: Check torch doc
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

def _corr2d(Z, W, S=(1,1), P=(0,0), D=(1,1)):
    """ 
    Convolution with stride and padding support
    Z: (N,C_in,H,W)
    W: (C_out,C_in,KH,KW)
    S: (SH,SW) stride
    P: (PH,PW) padding
    D: (DH,DW) dilation
    The reason we permute the dimensions is to make HWC of Z align with KKI of W as the last dim.

    >Note: This is the fastest conv in pure Numpy
    W = O*I*K*K -> K*K*I*O -> W.reshape(KH*KW*I, O) [----] # Each channel kernels represented by a single row
    Z = N*C*H*W -> N*H*W*C -> N*Z*ZH*ZW*KH*KW*C -> ZS.reshape(-1,KH*KW*inC)  # correpsonding small matrices & channels as vectors

    K = 10*4*3*3 -> innerDim:10*4*3*3=10,1,36 [----] # Each channel kernels represented by a single row
    Z = 10*4*8*8 -> 1*4*6*6*3*3 ->? 1         [-]  # The correpsonding small matrices & channels should be a vector
                                              [-]
                                              [-]
                                              [-]
    That's why the channels should be last after HxW

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

def conv2d(Z, W, mode:str="valid"): return _corr2d(*_pad(Z, np.flip(W), mode))
def corr2d(Z, W, mode:str="valid"): return _corr2d(*_pad(Z, W, mode))
def corr2d_backward(Z, W, TopGrad,  mode:str="valid"):
    WGrad = corr2d(Z.transpose(1,0,2,3), TopGrad.transpose(1,0,2,3)).transpose(1,0,2,3)
    ZGrad = np.flip(np.rot90(corr2d(TopGrad, W.transpose(1,0,2,3), "full"))).transpose(1,0,2,3)
    return WGrad , ZGrad

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
    """ 
    + TODO: We must suport stride (dilation!!)
    Performs the windowing, and padding if needed
    !Note: By default padding is set to False in most libs, Pytorch included
    if there are pixels left just drop them. We may need to reconsider.
    """
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
    outShape = (N,C,ZH//KH, ZW//KW, KH, KW)
    Zstrided = as_strided(Z, shape=outShape, strides=(Ns, Cs, Hs*KH, Ws*KW,Hs, Ws))
    return Zstrided.reshape(N,C,ZH//KH, ZW//KW, KH*KW) # reshape to flatten pool windows to 1D-vec

def maxpool2d(Z, K:tuple=(2,2)):
    """
    + TODO: We need to return indx of flattened version for faster backward to avoid loops
    """
    ZP = pool2d(Z, K)
    MxP = np.max(ZP, axis=(-1))
    Inx = np.argmax(ZP, axis=-1)
    return MxP, Inx

def unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
    # abhorent change this, to im2col
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

# Using ascontiguousarray to return new array avoid refrencing the old
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


def _lstm_cell(Z, H, C, W_hh, W_ih, B):
    i,f,g,o = np.split(Z@W_ih + H@W_hh + B[None,:], 4, axis=1) # Input, Forget,g (tanh-Activation) , Output
    i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f*c + i*g
    h_out = o * np.tanh(c_out)
    cache = i,f,o,g, c_out, C,x , h_out, Wx, Wh
    return h_out, c_out
def _lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        H[t,:,:] = h # Batch Comes second for contiguous memory :,:
    return H, c
def lstm_cell(x, prev_h, prev_c, Wx, Wh, b): # swapping o/g
    a = Z@W_ih + H@W_hh + b # (1, 4*hidden_dim) if b.shape dont match use this b[None,:]
    i,f,g,o = np.split(a, 4, axis=1) # Input, Forget,g (tanh-Activation) , Output
    i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    next_c = f * prev_c + i * g             # (1, hidden_dim)
    next_h = o * (np.tanh(next_c))          # (1, hidden_dim)
    cache = x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c
    return next_h, next_c, cache

def lstm(x, prev_h, prev_c, Wx, Wh, b):
    cache = []
    for i in range(x.shape[0]):     # 0 to seq_length-1
        next_h, next_c, next_cache = lstm_step_forward(x[i][None], prev_h, prev_c, Wx, Wh, b)
        prev_h = next_h
        prev_c = next_c
        cache.append(next_cache)
        if i > 0: h = np.append(h, next_h, axis=0)
        else: h = next_h
    return h, cache

def lstm_step_backward(dnext_h, dnext_c, cache):
    x, prev_h, prev_c, Wx, Wh, b, a, i, f, o, g, next_c = cache
    d1 = o * (1 - np.tanh(next_c) ** 2) * dnext_h + dnext_c
    dprev_c = f * d1
    dop = np.tanh(next_c) * dnext_h
    dfp = prev_c * d1
    dip = g * d1
    dgp = i * d1
    do = o * (1 - o) * dop
    df = f * (1 - f) * dfp
    di = i * (1 - i) * dip
    dg = (1 - g ** 2) * dgp
    da = np.concatenate((di, df, dg, do), axis=1)
    db = np.sum(da, axis=0)
    dx = da.dot(Wx.T)
    dprev_h = da.dot(Wh.T)
    dWx = x.T.dot(da)
    dWh = prev_h.T.dot(da)
    return dx, dprev_h, dprev_c, dWx, dWh, db

def lstm_backward(dh, cache):
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N, H = dh.shape
    dh_prev = 0
    dc_prev = 0
    for i in reversed(range(N)):
        dx_step, dh0_step, dc_step, dWx_step, dWh_step, db_step = lstm_step_backward(dh[i][None] + dh_prev, dc_prev, cache[i])
        dh_prev = dh0_step
        dc_prev = dc_step
        if i==N-1:
            dx = dx_step
            dWx = dWx_step
            dWh = dWh_step
            db = db_step
        else:
            dx = np.append(dx_step, dx, axis=0)
            dWx += dWx_step
            dWh += dWh_step
            db += db_step
    dh0 = dh0_step
    return dx, dh0, dWx, dWh, db



def normalize(Z): return (Z-np.mean(Z))/np.std(Z) # standardize really



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



# Didn't test, Make sure this is correct: Kapathry Lect 3 youtu.be/P6sfmUTpUmc & 4 youtu.be/q8SA3rM6ckI
def batchNormBackward(TopGrad, batchMean, batchVar, gama, beta):
    N = TopGrad.shape[0]
    Zhat = (TopGrad * gama)
    Zhat_batchMean = Zhat / np.sqrt(batchVar)
    Zhat_batchVar = np.sum(Zhat * (TopGrad * gama) * (-0.5) * ((batchVar + 1e-8) ** (-1.5)), axis=0)
    Zhat_batchVar_X = 2 * (TopGrad * gama - batchMean / N) / N
    Zgrad = Zhat_batchMean + Zhat_batchVar * Zhat_batchVar_X
    BNgainGrad = np.sum(TopGrad * (TopGrad - batchMean) / np.sqrt(batchVar), axis=0)
    BNbiasGrad = np.sum(TopGrad, axis=0)
    return Zgrad, BNgainGrad, BNbiasGrad
