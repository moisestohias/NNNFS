import numpy as np
import math
class Layer:
    def __init__(self): self.layers_name = self.__class__.__name__
    def __call__(self, x): return self.forward(x)
    def forward(self, x): raise NotImplementedError
    def backward(self, output_gradient, learning_rate): raise NotImplementedError


class RNN(Layer):
    def __init__(self, n_units, bptt_trunc=5, input_shape=None):
        self.activation = np.tanh
        self.input_shape = input_shape
        self.n_units = n_units
        self.bptt_trunc = bptt_trunc
        self.U = None # Weight of the input
        self.W = None # Weight of the previous state
        self.V = None # Weight of the output

        timesteps, input_dim = self.input_shape
        # Initialize the weights
        limit = 1 / math.sqrt(input_dim)
        self.U  = np.random.uniform(-limit, limit, (self.n_units, input_dim))
        limit = 1 / math.sqrt(self.n_units)
        self.V = np.random.uniform(-limit, limit, (input_dim, self.n_units))
        self.W  = np.random.uniform(-limit, limit, (self.n_units, self.n_units))

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.U.shape) + np.prod(self.V.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        batch_size, timesteps, input_dim = X.shape

        # Save these values for use in backprop.
        self.state_input = np.zeros((batch_size, timesteps, self.n_units))
        self.states = np.zeros((batch_size, timesteps+1, self.n_units))
        self.outputs = np.zeros((batch_size, timesteps, input_dim))

        # Set last time step to zero for calculation of the state_input at time step zero
        self.states[:, -1] = np.zeros((batch_size, self.n_units))
        for t in range(timesteps):
            # Input to state_t is the current input and output of previous states
            self.state_input[:, t] = X[:, t].dot(self.U.T) + self.states[:, t-1].dot(self.W.T)
            self.states[:, t] = self.activation(self.state_input[:, t])
            self.outputs[:, t] = self.states[:, t].dot(self.V.T)

        return self.outputs

    def backward_pass(self, accum_grad, LR):
        _, timesteps, _ = accum_grad.shape

        # Variables where we save the accumulated gradient w.r.t each parameter
        grad_U = np.zeros_like(self.U)
        grad_V = np.zeros_like(self.V)
        grad_W = np.zeros_like(self.W)
        # The gradient w.r.t the layer input.
        # Will be passed on to the previous layer in the network
        accum_grad_next = np.zeros_like(accum_grad)

        # Back Propagation Through Time
        for t in reversed(range(timesteps)):
            # Update gradient w.r.t V at time step t
            grad_V += accum_grad[:, t].T.dot(self.states[:, t])
            # Calculate the gradient w.r.t the state input
            grad_wrt_state = accum_grad[:, t].dot(self.V) * self.activation.gradient(self.state_input[:, t])
            # Gradient w.r.t the layer input
            accum_grad_next[:, t] = grad_wrt_state.dot(self.U)
            # Update gradient w.r.t W and U by backprop. from time step t for at most
            # self.bptt_trunc number of time steps
            for t_ in reversed(np.arange(max(0, t - self.bptt_trunc), t+1)):
                grad_U += grad_wrt_state.T.dot(self.layer_input[:, t_])
                grad_W += grad_wrt_state.T.dot(self.states[:, t_-1])
                # Calculate gradient w.r.t previous state
                grad_wrt_state = grad_wrt_state.dot(self.W) * self.activation.gradient(self.state_input[:, t_-1])

        # Update weights
        self.U = self.U - grad_U*LR
        self.V = self.V - grad_V*LR
        self.W = self.W - grad_W*LR


"""
xs = np.arange(-10, 10, 0.01, dtype=np.float32) # 2000 sample, 50 W, 40 WS
InDim = 40
strided_xs = np.lib.stride_tricks.sliding_window_view(xs, InDim)
strided_xs = np.expand_dims(strided_xs, axis=0)
ys = np.sin(strided_xs)

rnn = RNN(InDim, 20, (30,InDim))
out = rnn.forward_pass(strided_xs)
print(out.shape)

"""

# =================================================
# # RNN: (N,L) -> (N, H) -> (N, O)
# N, L, H, O = 2, 10, 5, 3
# Z = np.random.randn(N,L)
# Hid = np.random.randn(H)
# Whh = np.random.randn(H,H)
# Wih = np.random.randn(L,H) # (H,L) * H -> H
# Why = np.random.randn(H,O)
# Bh = np.random.randn(H)
# By = np.random.randn(O)


def rnn(Z, H, W_hh, W_ih, W_hy, Bh, By, actFun=tanh):
    zh = H.dot(W_hh) + Z.dot(W_ih) + Bh
    ht = actFun(zh)
    yt = ht.dot(W_hy) + By
    out = actFun(yt)
    return ht, yt, zh, out

def rnn_prime(TopGrad, Z, ht, yt, zh, W_ih, W_hh, W_hy, Bh, By, actFunPrime=tanh_prime):
    TopGrad = TopGrad * actFunPrime(yt)
    Z_hyGrad, W_hyGrad, B_hyGrad = backward_affine_transform(TopGrad, ht, W_hy)
    yt_Grad = Z_hyGrad * actFunPrime(zh)
    Z_hhgrad, W_hhGrad, B_hhGrad = backward_affine_transform(Zgrad,ht,W_hh)
    Z_ihgrad, W_ihGrad, B_ihGrad = backward_affine_transform(Zgrad,Z,W_ih)
    BGrad = TopGrad.sum(axis=0)
    return ButtomGrad, Z_hhgrad, W_hhGrad, B_hhGrad, Z_ihgrad, W_ihGrad, B_ihGrad

# Z: (2,6), W: (6,4), B: (4) => A: (2, 4)
# A: (2,4)
# # Zgrad: (2, 6), WGrad: (6, 4) , BGrad:  (4,)
# Hid, Yt, zh, A = rnn(Z, Hid, Whh, Wih, Why, Bh, By)
# # print(H.shape, Yt.shape, A.shape)
# # print(TopGrad.shape, Z.shape, Hid.shape, Yt.shape, zh.shape, Whh.shape, Wih.shape, Why.shape, Bh.shape, By.shape, tanh_prime)
# TopGrad = np.ones_like(A)
# rnn_prime(TopGrad, Z, Hid, Yt, zh, Wih, Whh, Why, Bh, By, tanh_prime)


# Z = np.random.randn(1,6)
# W = np.random.randn(6,4)
# B = np.random.randn(4)
# A = affine_transform(Z,W,B)
# Zgrad, WGrad, BGrad = backward_affine_transform(np.ones_like(A), Z, W)
# print(Zgrad.shape, WGrad.shape, BGrad.shape)
