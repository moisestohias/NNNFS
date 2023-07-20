# functional.py
import numpy as np
from math import ceil, floor
as_strided = np.lib.stride_tricks.as_strided


# quantdare.com/implementing-a-rnn-with-numpy/

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



