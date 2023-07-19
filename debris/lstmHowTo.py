# lstmHowTo.py

import numpy as np
def sigmoid(x): return np.reciprocal((1.0+np.exp(-x)))
def sigmoid_prime(x): s = np.reciprocal((1.0+np.exp(-x))); return s * (1 - s) # σ(x)*(1-σ(x))
def tanh(x): return np.tanh(x) # or 2.0*(σ((2.0 * x)))-1.0
def tanh_prime(x): return 1 - np.tanh(x) ** 2

"""
josehoras.github.io/lstm-pure-python/
github.com/josehoras/LSTM-Frameworks
github.com/eliben/deep-learning-samples/tree/master/min-char-rnn
https://skillsmatter.com/skillscasts/6611-visualizing-and-understanding-recurrent-networks
bingyuzhou.github.io/deep-learning/2018/03/19/LSTM/
pub.towardsai.net/building-a-lstm-from-scratch-in-python-1dedd89de8fe
stackoverflow.com/questions/41555576/lstm-rnn-backpropagation
github.com/BingyuZhou?language=&page=1&q=&sort=&tab=repositories

RNN
https://github.com/JY-Yoon/RNN-Implementation-using-NumPy
https://karpathy.github.io/2015/05/21/rnn-effectiveness/

The tanh component of the LSTM cell is denoted by g because it is the activation function that computes the candidate values for the cell state1. It is also applied to the final cell state to produce the hidden state2. The tanh function helps to keep the values between -1 and 1 and avoid exploding gradients2.

# Input, Forget, G (tanh) , Output
# Input: Input gate scales the candidate values which are the new information that can candidate to be added to the cell state after being . Bascially the input gate decides what to keep or to loose from the candidate values by scalling their values.
+ Forget: What information to keep or to Forget from the previous cell state.
+ G:  The tanh component of the LSTM cell (aka candidate values) is denoted by g because instead of c because c is already used for the cell state, which is the memory of the LSTM network.
!Note: tanh is the activation function that computes the candidate values for the cell state. It's used to keep the values between -1 and 1 and avoid exploding gradients.
!Note: It is also applied to the final cell state to produce the hidden state.
+ Output: The output gate decides what information should exit or suppressed.
"""

"""
def lstm_cell(x, h, c, W_hh, W_ih, b):
    i,f,g,o = np.split(W_ih@x + W_hh@h + b, 4)
    i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f*c + i*g,
    h_out = o * np.tanh(c_out)
    return h_out, c_out

def lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        H[t,:,:] = h # Batch Comes second for contiguous memory :,:
    return H, c

sizes = (20,), (100,), (100,), (400,100), (400, 20), (400,)
Z, H, C, W_hh, W_ih, B = [np.random.randn(*s) for s in sizes]
# for i in (Z, H, C, W_hh, W_ih, B ): print(i.shape)
lstm_cell(Z, H, C, W_hh, W_ih, B)

"""
# Re-Write lstm_cell and lstm function to return the cache, to be able to compute the gradient
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

    Input:
        + Z: Input of shape (N, D): N: batch, D Size 
        + H: Previous Hidden state of shape (N, H) 
        + C: Previous Cell state of shape (N, H) 
        + W_hh: Weight of the Hidden state of shape (H, 4H) 
        + W_ih: Weight of the Cell state of shape (D, 4H) 
        + db: Gradient of biases, of shape (4H)
    Output:
        + h_out: Current hidden state of shape (N, H)
        + c_out: Current hidden state of shape (N, H)
        + cache: A set of everything needed for the backward pass
    """
    i,f,g,o = np.split(Z@W_ih + H@W_hh + b[None,:], 4, axis=1)
    i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f*C + i*g
    h_out = o * np.tanh(c_out)
    cache = i,f,o,g,c_out, C,Z, H, W_ih, W_hh
    return h_out, c_out, cache

def lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
    cache = []
    for t in range(X.shape[0]):
        h, c, cache_ = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        cache.append(cache_)
        H[t,:,:] = h
    return H, c, cache

       # (2,20,), (2,100,), (2,100,), (100,400), (20, 400), (400,)
sizes = (50,80,20), (80,100), (80,100), (100,400), (20, 400), (400,)
Z, H, C, W_hh, W_ih, B = [np.random.randn(*s) for s in sizes]
H, c, cache = lstm(Z, H, C, W_hh, W_ih, B)
print(c.shape)


def lstm_step_backward(dnext_h, dnext_c, cache):
    """
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

# the backward pass of a lstm cell


def lstm_prime(X, H, C, dH, dC, W_hh, W_ih, b):
    dW_hh, dW_ih, dB = np.zeros_like(W_hh), np.zeros_like(W_ih), np.zeros_like(b)
    dX = np.zeros_like(X)
    dH_prev = np.zeros_like(H[0])
    dC_prev = np.zeros_like(C[0])

    for t in reversed(range(X.shape[0])):
        dH_t = dH[t] + dH_prev
        dC_t = dC_prev + dH_t * (1 - np.tanh(C[t])**2) * H[t]
        dO_t = dH_t * np.tanh(C[t]) * sigmoid_prime(H[t])
        dG_t = dC_t * sigmoid(H[t]) * (1 - np.tanh(C[t])**2)
        dI_t = dC_t * np.tanh(G[t]) * sigmoid_prime(I[t])
        dF_t = dC_t * C[t-1] * sigmoid_prime(F[t])

        dZ_t = np.concatenate((dI_t, dF_t, dG_t, dO_t), axis=1)
        dW_ih += X[t].T @ dZ_t
        dW_hh += H[t-1].T @ dZ_t
        dB += np.sum(dZ_t, axis=0)

        dX[t] = dZ_t @ W_ih.T
        dH_prev = dZ_t @ W_hh.T
        dC_prev = dC_t

    return dX, dH_prev, dC_prev, dW_hh, dW_ih, dB
