import numpy as np
from math import ceil, floor
as_strided = np.lib.stride_tricks.as_strided


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

def lstm_prime(dH, dc, cache):
    dx, dh0, dc0, dWx, dWh, db = None, None, None, None, None, None

    # Unpack the cache
    X, H, C, Wx, Wh, b, Z, I, F, G, O = cache

    # Initialize gradients
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    db = np.zeros_like(b)
    dX = np.zeros_like(X)
    dH_prev = np.zeros_like(H[0])
    dC_prev = np.zeros_like(C[0])

    # Loop through the sequence backwards
    for t in reversed(range(len(X))):
        # Compute gradients for this timestep
        dh = dH[t] + dH_prev
        dc = dc + dC_prev

        do = dh * np.tanh(C[t])
        dC = dh * O[t] * (1 - np.tanh(C[t])**2) + dc
        di = dC * G[t]
        dg = dC * I[t]
        df = dC * C[t-1]

        # Compute gates gradients
        dZ = np.zeros_like(Z[t])
        dZ[:, :I.shape[1]] = sigmoid(I[t]) * (1 - sigmoid(I[t])) * di
        dZ[:, I.shape[1]:2*I.shape[1]] = sigmoid(F[t]) * (1 - sigmoid(F[t])) * df
        dZ[:, 2*I.shape[1]:3*I.shape[1]] = (1 - np.tanh(G[t])**2) * dg
        dZ[:, 3*I.shape[1]:] = sigmoid(O[t]) * (1 - sigmoid(O[t])) * do

        # Compute gradients for inputs to gates
        dWx += X[t].T @ dZ
        dWh += H[t-1].T @ dZ
        db += np.sum(dZ, axis=0)

        # Compute gradients for inputs to LSTM cell
        dX[t] = dZ @ Wx.T
        dH_prev = dZ @ Wh.T
        dC_prev = dC * F[t]

    return dX, dH_prev, dC_prev, dWx, dWh, db
