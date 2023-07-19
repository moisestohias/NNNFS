import numpy as np
from math import ceil, floor
as_strided = np.lib.stride_tricks.as_strided

def sigmoid(x): return np.reciprocal((1.0+np.exp(-x)))
def sigmoid_prime(x): s = np.reciprocal((1.0+np.exp(-x))); return s * (1 - s) # σ(
def tanh(x): return np.tanh(x)
def tanh_prime(x): return 1 - np.tanh(x) ** 2

def lstm_cell(Z, H, C, W_hh, W_ih, b):
    i,f,g,o = np.split(Z@W_ih + H@W_hh + b[None,:], 4, axis=1)
    i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f*C + i*g
    h_out = o * np.tanh(c_out)
    cache = i,f,o,g,c_out, C,Z, H, W_ih, W_hh
    return h_out, c_out, cache

def lstm_cell_backward_(dnext_h, dnext_c, cache):
  # Unpack the cache
  i,f,o,g,c_next,c_prev,z,h_prev,w_ih,w_hh = cache
  
  # Get the shapes of the tensors
  N,H = dnext_h.shape
  K = w_ih.shape[0] // 4

  # Initialize the gradients for input, weight and bias
  dx = np.zeros((N,K))
  dw_ih = np.zeros_like(w_ih)
  dw_hh = np.zeros_like(w_hh)
  db = np.zeros((4*H,))
  
  # Compute the gradients for the gates and cell state
  do = dnext_h * np.tanh(c_next)
  dnext_c += dnext_h * o * (1 - np.tanh(c_next)**2)
  df = dnext_c * c_prev
  dc_prev = dnext_c * f
  di = dnext_c * g
  dg = dnext_c * i
  
  # Apply the activation function derivatives
  do *= o * (1 - o)
  df *= f * (1 - f)
  di *= i * (1 - i)
  dg *= (1 - g**2)
  
  # Concatenate the gate gradients
  dz = np.concatenate((di,df,dg,do), axis=1)
  
  # Update the input gradient
  # W_hh: Weight of the Hidden state of shape (H, 4H)
  # W_ih: Weight of the Cell state of shape   (D, 4H)
  # here we have shape miss match. Where dz: (N, 4*H), w_ih: (D, 4*H)
  # dz.shape:(1, 4*H) w_ih.T.shape:(4*H, D)
  dx += dz.dot(w_ih.T)
  
  # Update the weight gradients
  dw_ih += z.T.dot(dz)
  dw_hh += h_prev.T.dot(dz)
  
  # Update the bias gradient
  db += np.sum(dz, axis=0)
  
  # Update the previous hidden state gradient
  dh_prev = dz.dot(w_hh.T)
  
  return dx,dw_ih,dw_hh,db,dh_prev,dc_prev

def lstm_cell_backward(dnext_h, dnext_c, cache):
  # Unpack the cache
  i,f,o,g,c_out, C,Z, H, W_ih, W_hh
  i,f,o,g,c_next,c_prev,z,h_prev,x,w_ih,w_hh,b_ih,b_hh = cache

  # Get the shapes of the tensors
  N,H = dnext_h.shape
  K = w_ih.shape[0] // 4

  # Initialize the gradients for input, weight and bias
  dx = np.zeros((N,K))
  dw_ih = np.zeros_like(w_ih)
  dw_hh = np.zeros_like(w_hh)
  db_ih = np.zeros_like(b_ih)
  db_hh = np.zeros_like(b_hh)

  # Compute the gradients for the gates and cell state
  do = dnext_h * np.tanh(c_next)
  dnext_c += dnext_h * o * (1 - np.tanh(c_next)**2)
  df = dnext_c * c_prev
  dc_prev = dnext_c * f
  di = dnext_c * g
  dg = dnext_c * i

  # Apply the activation function derivatives
  do *= o * (1 - o)
  df *= f * (1 - f)
  di *= i * (1 - i)
  dg *= (1 - g**2)

  # Concatenate the gate gradients
  dz = np.concatenate((di,df,dg,do), axis=1)

  # Update the input gradient
  dx = dz.dot(w_ih.T)

  # Update the weight gradients
  dw_ih = x.T.dot(dz)
  dw_hh = h_prev.T.dot(dz)

  # Update the bias gradients
  db_ih = np.sum(dz, axis=0)
  db_hh = np.sum(dz, axis=0)

  # Update the previous hidden state gradient
  dh_prev = dz.dot(w_hh.T)

  return dx, dh_prev, dc_prev, dw_ih, dw_hh, db_ih, db_hh

# Z: Input of shape                         (N, D)
# H: Previous Hidden state of shape         (N, H)
# C: Previous Cell state of shape           (N, H)
# W_hh: Weight of the Hidden state of shape (H, 4H)
# W_ih: Weight of the Cell state of shape   (D, 4H)
# db: Gradient of biases, of shape          (4H)
N, D = 1, 10
H = 5

Z = np.random.randn(N, D)
H0 = np.random.randn(N, H)
C = np.random.randn(N, H)
W_hh = np.random.randn(H, 4*H)
W_ih = np.random.randn(D, 4*H)
db = np.random.randn(4*H)
dnext_h, dnext_c = np.ones((N,H)), np.ones((N,H))
h_out, c_out, cache = lstm_cell(Z, H0, C, W_hh, W_ih, db)
# h_out, c_out, cache = lstm_cell(Z, H0, C, W_hh.T, W_ih.T, db)
# print(h_out.shape, c_out.shape)


lstm_cell_backward(dnext_h, dnext_c, cache)