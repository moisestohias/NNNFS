# functional.py
import numpy as np
from math import ceil, floor
as_strided = np.lib.stride_tricks.as_strided


def _pad(Z: np.ndarray, K: np.ndarray, mode: str="valid") -> np.ndarray:
    """ Check arguments and pad for conv/corr """
    if mode not in ["full", "same", "valid"]: raise ValueError("mode must be one of ['full', 'same', 'valid']")
    if Z.ndim != K.ndim: raise ValueError("Z and K must have the same number of dimensions")
    if Z.size == 0 or K.size == 0: raise ValueError(f"Zero-size arrays not supported in convolutions.")

    # InputLargerThanKernel = all(s1 >= s2 for s1, s2 in zip(Z.shape[1:], K.shape[1:]))
    # if not InputLargerThanKernel: raise ValueError(f"Input must be larger than the Kernel in every dimension, except Depth. Got Input {Z.shape[1:]}, Kernel: {K.shape[1:]}", )
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

def _corr2d(Z, W):
    """ Fastest conv in pure Numpy other implmenetation are found at the bottom
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

# Pooling
def pool1d(Z, K:int=2):
    N, C, W = Z.shape #NCHW
    Ns, Cs, Ws = Z.strides
    EdgeW = W%K # How many pixels left on the edge
    if  EdgeW!=0: # If there are pixelx left we need to pad
        PadW = K-EdgeW
        PadLeft, PadRight = ceil(PadW/2), floor(PadW/2)
        Z = np.pad(Z, ((0,0),(0,0), (PadLeft, PadRight)))
        N, C, W = Z.shape #NCHW
        Ns, Cs, Ws = Z.strides
    return as_strided(Z, shape=(N,C, W//K, K), strides=(Ns, Cs, Ws*K , Ws) )

def pool2d(Z, K:tuple=(2,2)):
    """ performs the windowing, and padding if needed
    !Note: most implementations including Pytorch don't pad,
    if there are pixels left just drop them. We may need to reconsider.
    """
    KH, KW = K  # Kernel Height & Width
    N, C, ZH, ZW = Z.shape # Input: NCHW Batch, Channels, Height, Width
    Ns, Cs, Hs, Ws = Z.strides
    EdgeH, EdgeW = ZH%KH, ZW%KW # How many pixels left on the edge
    if (EdgeH!=0 or EdgeW!=0): # If there are pixels left we need to pad
        PadH, PadW = KH-EdgeH, KW-EdgeW
        PadTop, PadBottom = ceil(PadH/2), floor(PadH/2)
        PadLeft, PadRight = ceil(PadW/2), floor(PadW/2)
        Z = np.pad(Z, ((0,0),(0,0), (PadTop, PadBottom), (PadLeft, PadRight)))
        N, C, ZH, ZW = Z.shape #NCHW
        Ns, Cs, Hs, Ws = Z.strides
    Zstrided = as_strided(Z, shape=(N,C,ZH//KH, ZW//KW, KH, KW), strides=(Ns, Cs, Hs*KH, Ws*KW,Hs, Ws))
    return Zstrided.reshape(N,C,ZH//KH, ZW//KW, KH*KW) # reshape to flatten pooling windows to be 1D-vector


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


# def unpool1d(ZP, K): np.repeat(Z, K, axis=-1)
# def unpool2d(ZPooled, K): return np.repeat(np.repeat(ZPooled, K[0], axis=-2), K[1], axis=-1)
# The as_strided version of the unpoolxd is faster for larger input/kernel
#!Note: using ascontiguousarray to return new array avoid refrencing the old
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
def maxpool2d(Z, K:tuple=(2,2)):
    ZP = pool2d(Z, K)
    MxP = np.max(ZP, axis=(-1))
    Inx = np.argmax(ZP, axis=-1)
    return MxP, Inx

def normalize(Z): return (Z-np.mean(Z))/np.std(Z) # standardize really

# Losses
def cross_entropy(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return - y * np.log(p) - (1 - y) * np.log(1 - p)
def cross_entropy_prim(y, p): p = np.clip(p, 1e-15, 1 - 1e-15); return  - (y / p) + (1 - y) / (1 - p)
def binary_cross_entropy(y, p): return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))
def binary_cross_entropy_prime(y, p): return ((1 - y) / (1 - p) - y / p) / np.size(y)
def mse(y, p): return 0.5*np.power((y-p), 2).mean()
def mse_prime(y, p): return (p-y).mean()

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
def mish(x): raise NotImplementedError #arxiv.org/vc/arxiv/papers/1908/1908.08681v2.pdf
def mish_prime(x):  raise NotImplementedError
def hard_mish(x): return np.minimum(2., relu(x + 2.)) * 0.5 * x # github.com/cpuimage/HardMish
def hard_mish_prime(x): raise NotImplementedError #Yet
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
The approximate derivative of the GELU activation function is:
f'(x) = 0.5 * x * (1 + tanh(x/2))
The derivative of the GELU activation function is given by:
f'(x) = 0.5 * (1 + erf(x/sqrt(2))) + 0.5 * 2 * exp(-x^2/2) / sqrt(2 * pi)
"""



def rnn(Z, H, W_hh, W_ih, Bh, By, actFun=tanh):
    """
    Parameters
    ----------
        + Z: ndarray: input
        + H: ndarray: hidden state
        + W_hh ndarray: hh: Weight of the hidden state
        + W_ih ndarray: ih: Weight of the input
        + Bh: ndarray: biase of the hidden state
        + By: ndarray: biase of the output
    Returns
    -------
    ht: ndarray: hidden state
    out : ndarray: output
    """
    zh = W_hh.dot(h) + W_ih.dot(i) + Bh
    ht = actFun(zh)
    yt = W_hh.dot(h) + by
    out = actFun(W_hh.dot(yt) + By)
    return ht, output

def rnn_prime(TopGrad, Z, H, W_ih, W_hh, Bh, By, actFun=tanh):
    """
    Parameters
    ----------
        + Z: ndarray: input
        + H: ndarray: hidden state
        + W_hh ndarray: hh: Weight of the hidden state
        + W_ih ndarray: ih: Weight of the input
        + Bh: ndarray: biase of the hidden state
        + By: ndarray: biase of the output
    Returns
    -------
    ht: ndarray: hidden state
    out : ndarray: output
    """
    return ht, output
    Z
    out = actFun(W_hh.dot(yt) + By)
    yt = W_hh.dot(h) + by
    ht = actFun(zh)
    zh = W_hh.dot(h) + W_ih.dot(i) + Bh



def lstm_cell(Z, H, C, W_hh, W_ih, b):
    """
      Input:
        N: Batch, D Input Size, H Hidden State
        + Z: Input of shape                         (N, D)
        + H: Previous Hidden state of shape         (N, H)
        + C: Previous Cell state of shape           (N, H)
        + W_hh: Weight of the Hidden state of shape (H, 4H)
        + W_ih: Weight of the Cell state of shape   (D, 4H)
        + db: Gradient of biases, of shape          (4H)
    """
    i,f,g,o = np.split(Z@W_ih + H@W_hh + b[None,:], 4, axis=1)
    i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f*C + i*g
    h_out = o * np.tanh(c_out)
    cache = i,f,o,g,c_out, C,Z, H, W_ih, W_hh
    return h_out, c_out, cache

def lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        H[t,:,:] = h # Batch Comes second for contiguous memory :,:
    return H, c



def lstm_step_backward(dnext_h, dnext_c, cache):
    """
    Backward pass for a single timestep of an LSTM.

    Inputs:
    - dnext_h: Gradients of next hidden state, of shape (N, H)
    - dnext_c: Gradients of next cell state, of shape (N, H)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient of input data, of shape (N, D)
    - dprev_h: Gradient of previous hidden state, of shape (N, H)
    - dprev_c: Gradient of previous cell state, of shape (N, H)
    - dWx: Gradient of input-to-hidden weights, of shape (D, 4H)
    - dWh: Gradient of hidden-to-hidden weights, of shape (H, 4H)
    - db: Gradient of biases, of shape (4H,)
    """
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None

    i,f,o,g,next_c, prev_c,x, prev_h, Wx, Wh = cache
    N,D = x.shape
    dnext_h_next_c = (1-np.tanh(next_c)**2)*o*dnext_h
    dprev_c_next_c = f*dnext_c
    dprev_c_next_h = f*dnext_h_next_c

    dprev_c = dprev_c_next_c+dprev_c_next_h

    dai_next_c = i*(1-i)*g*(dnext_c+dnext_h_next_c)
    daf_next_c = f*(1-f)*prev_c*(dnext_c+dnext_h_next_c)
    dao_next_h = o*(1-o)*np.tanh(next_c)*dnext_h
    dag_next_c = (1-g**2)*i*(dnext_c+dnext_h_next_c)

    stack = np.concatenate((x, prev_h), axis=1)
    d_activation = np.concatenate((dai_next_c, daf_next_c, dao_next_h, dag_next_c), axis=1)
    dW = np.dot(stack.T,d_activation)
    dWx = dW[:D,:]
    dWh = dW[D:,:]
    db = np.sum(d_activation,axis=0)

    W = np.concatenate((Wx, Wh), axis=0)
    dxh = np.dot(d_activation, W.T)
    dx = dxh[:,:D]
    dprev_h = dxh[:,D:]

    return dx, dprev_h, dprev_c, dWx, dWh, db

"""
InLen  = 20
OutLen = 100
MBS    = 80
SeqLen = 50
X = np.random.randn(SeqLen,MBS,InLen).astype(np.float32) # MBS comes after the seq length bc WhyMBSComessecond
h0 = np.random.randn(MBS,OutLen).astype(np.float32)
c0 = np.random.randn(MBS,OutLen).astype(np.float32)
model = nn.LSTM(20, OutLen, num_layers = 1)
H_, (hn_, cn_) = model(torch.tensor(X), (torch.tensor(h0)[None,:,:], torch.tensor(c0)[None,:,:]))

H, cn = lstm(X, h0, c0,
             model.weight_hh_l0.detach().numpy().T,
             model.weight_ih_l0.detach().numpy().T,
             (model.bias_hh_l0 + model.bias_ih_l0).detach().numpy())


print(np.linalg.norm(H - H_.detach().numpy()),
      np.linalg.norm(cn - cn_[0].detach().numpy()))
"""


Z = np.random.randn(1,6)
W = np.random.randn(6,3)
B = np.random.randn(3)

def rnn(Z, H, W_hh, W_ih, Bh, By, actFun=tanh):
    zh = W_hh.dot(h) + W_ih.dot(i) + Bh
    ht = actFun(zh)
    yt = W_hh.dot(h) + by
    out = actFun(W_hh.dot(yt) + By)
    return ht, output



