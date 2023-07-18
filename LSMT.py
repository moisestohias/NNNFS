
"""
https://blog.varunajayasiri.com/numpy_lstm.html
https://quantdare.com/implementing-a-rnn-with-numpy/
https://mattgorb.github.io/lstm_numpy
https://github.com/nicodjimenez/lstm/blob/master/test.py
https://github.com/CaptainE/RNN-LSTM-in-numpy
https://towardsdatascience.com/implementing-recurrent-neural-network-using-numpy-c359a0a68a67
https://cs231n.github.io/neural-networks-case-study/#grad
"""
import numpy as np

def sigmoid(x): return np.reciprocal((1.0+np.exp(-x)))
def sigmoid_prime(x): s = np.reciprocal((1.0+np.exp(-x))); return s * (1 - s) # σ(x)*(1-σ(
def tanh(x): return np.tanh(x) # or 2.0*(σ((2.0 * x)))-1.0
def tanh_prime(x): return 1 - np.tanh(x) ** 2

def lstm_cell(Z, H, C, W_hh, W_ih, b):
    i,f,g,o = np.split(Z@W_ih + H@W_hh + b[None,:], 4, axis=1)
    i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f*C + i*g
    h_out = o * np.tanh(c_out)
    cache = i,f,o,g,c_out, C,Z, H, W_ih, W_hh
    return h_out, c_out, cache


def lstm_step_backward(HP, CP, cache):
    dx, dprev_h, dprev_c, dWx, dWh, db = None, None, None, None, None, None
    i,f,o,g,next_c, prev_c,x, prev_h, Wx, Wh = cache
    N,D = x.shape
    HP_next_c = (1-np.tanh(next_c)**2)*o*HP
    dprev_c_next_c = f*CP
    dprev_c_next_h = f*HP_next_c

    dprev_c = dprev_c_next_c+dprev_c_next_h

    dai_next_c = i*(1-i)*g*(CP+HP_next_c)
    daf_next_c = f*(1-f)*prev_c*(CP+HP_next_c)
    dao_next_h = o*(1-o)*np.tanh(next_c)*HP
    dag_next_c = (1-g**2)*i*(CP+HP_next_c)

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

def lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        H[t,:,:] = h # Batch Comes second for contiguous memory :,:
    return H, c


"""
Gradient must be calculated with respect:
	+ time
	+ weights:
		+ hidden state weights
		+ input weights
	+ input

Put in the forward pass to help you remenber/understand
W_ih.shape, x.shape, (W_ih@x).shape # (4*M, N) (N,) (4*M,)
W_hh.shape, h.shape, (W_hh@h).shape # (4*M, M) (M,) (4*M,)
(W_ih@x + W_hh@h + b).shape         # (4*M,)
i.shape,f.shape,g.shape,o.shape 		# (M,) (M,) (M,) (M,)

"""
class Layer:
	def __init__(self): self.layers_name = self.__class__.__name__
	def __call__(self, x): return self.forward(x)
	def forward(self, x): raise NotImplementedError
	def backward(self, output_gradient, learning_rate): raise NotImplementedError



class LSTM(Layer):
	def __init__(self, input_shape, output_shape):
		self.input_shape = input_shape
		self.output_shape = output_shape
		self.W_hh = np.random.randn(output_shape*4, output_shape)
		self.W_ih = np.random.randn(output_shape*4, input_shape)
		self.W_hh_grad = np.zeros_like(self.W_hh)
		self.W_ih_grad = np.zeros_like(self.W_ih)
		self.b = np.random.randn(1, output_shape)
		self.h = np.zeros((1, output_shape))
		self.c = np.zeros((1, output_shape))


	def forward(self,x):
		self.input = x
		print(self.W_ih.shape, x.shape)
		raise SystemExit
		i,f,g,o = np.split(self.W_ih@x + self.W_hh@self.h + self.b, 4)
		i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
		self.i, self.f, self.g, self.o = i,f,g,o # save for the backpass
		self.c_out = f*c + i*g
		self.h_out = o * np.tanh(self.c_out)
		return self.h_out, self.c_out

	def backward(self, CS_grad, HS_grad, LR): # Nx1
		# top dif back-propagating
		dSC= self.o * HS_grad + CS_grad
		do = self.s * HS_grad
		di = self.g * dSC
		dg = self.i * dSC
		df = self.s_prev * dSC

		# diffs w.r.t. vector inside sigm/tanh function
		di_input = sigmoid_prime(self.i) * di
		df_input = sigmoid_prime(self.f) * df
		do_input = sigmoid_prime(self.o) * do
		dg_input = tanh_prime(self.g) * dg

		# diffs w.r.t. inputs
		self.wi_diff += np.outer(di_input, self.input)
		self.wf_diff += np.outer(df_input, self.input)
		self.wo_diff += np.outer(do_input, self.input)
		self.wg_diff += np.outer(dg_input, self.input)
		self.bi_diff += di_input
		self.bf_diff += df_input
		self.bo_diff += do_input
		self.bg_diff += dg_input

def lstm_cell(x, h, c, W_hh, W_ih, b):
    i,f,g,o = np.split(x@W_ih + h@W_hh + b[None,:], 4, axis=1)
    i,f,g,o = sigmoid(i), sigmoid(f), np.tanh(g), sigmoid(o)
    c_out = f*c + i*g
    h_out = o * np.tanh(c_out)
    return h_out, c_out

def lstm(X, h, c, W_hh, W_ih, b):
    H = np.zeros((X.shape[0], X.shape[1], h.shape[1]))
    for t in range(X.shape[0]):
        h, c = lstm_cell(X[t], h, c, W_hh, W_ih, b)
        H[t,:,:] = h
    return H, c


OutNodes = 100
TimeSeq = 20
MBS = 80
InputLen = 50
X = np.random.randn()
X = np.random.randn(TimeSeq, MBS, InputLen).astype(np.float32)
h0 = np.random.randn(MBS,OutNodes).astype(np.float32)
c0 = np.random.randn(MBS,OutNodes).astype(np.float32)
lstm(H, C)

# xs = np.arange(-10, 10, 0.01, dtype=np.float32) # 2000 sample, 50 W, 40 WS
# InDim = 40
# strided_xs = np.lib.stride_tricks.sliding_window_view(xs, InDim)
# X = np.expand_dims(strided_xs, axis=0)
# Y = np.sin(X)
# print(X.shape, Y.shape)
# np.random.seed(12)

# rnn = LSTM(10, 10)
# x = np.random.randn(1,10)
# h, s = rnn(x)
# print(h.shape, s.shape)

