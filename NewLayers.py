from math import ceil, floor
import numpy as np
from functional import conv2d, corr2d, maxpool2d, unmaxpool2d
from numpy.lib.stride_tricks import as_strided

"""
LR: LearningRate
ZN, ZC, ZW,ZH = Batch, Channel, Height, Width
KW,KH = KernelHeight, KernelWidth
"""

class Layer:
  def __init__(self): self.layer_name = self.__class__.__name__
  def __call__(self, *a, **kq): return self.forward(*a, **kq)
  def __repr__(self): return f"{self.layer_name}{self.input_shape, self.output_shape}"
  def forward(self, x): raise NotImplementedError
  def backward(self, output_gradient, LR): raise NotImplementedError

class Dense(Layer):
  def __init__(self, input_shape, output_nodes):
    self.MBS, self.InputNodes = input_shape   #NL Batch,InputLen
    self.input_shape = input_shape   #NL Batch,InputLen
    self.output_nodes = output_nodes #NM Batch,OutLen
    self.output_shape = self.MBS, output_nodes #NM Batch,OutLen
    lim = 1/np.sqrt(self.InputNodes)
    self.weights = np.random.uniform(-lim, lim, (self.InputNodes, self.output_nodes))
    self.biases = np.random.uniform(-1, 1, (1, self.output_nodes))
    self.layer_name = self.__class__.__name__
  def __repr__(self): return f"{self.layer_name}{self.input_shape, self.output_nodes}"

  def forward(self, x):
    self.input = x # Save for the back-pass x: Nx1
    return x.dot(self.weights) + self.biases

  def backward(self, top_grad, LR): # Nx1
    bias_grad = top_grad.sum(axis=0)
    weight_grad = self.input.T.dot(top_grad) # Nx1•1xM = NxM
    input_grad = top_grad.dot(self.weights.T) # MxN•Nx1 = Mx1
    self.biases -= bias_grad *LR
    self.weights -= weight_grad*LR
    return input_grad


class Conv2d(Layer):
  def __init__(self, input_shape, OutCh, K, stride=1):
    K = K if (isinstance(K, tuple) and len(K)==2) else (K, K)
    self.input_shape = input_shape
    self.OutCh = OutCh # Output channels aka depth
    self.MBS, self.InCh, self.InH, self.InW = input_shape
    self.KH, self.KW = K # KernelSize
    self.stride = stride
    self.output_shape = self.MBS, OutCh, int((self.InH-self.KH)/stride)+1, int((self.InW-self.KW)/stride)+1
    self.weights = np.random.randn(self.OutCh, self.InCh, self.KH, self.KW)
    self.biases = np.random.randn(*self.output_shape[1:])
    self.layer_name = self.__class__.__name__
  def __repr__(self): return f"{self.layer_name}{self.weights.shape}"

  def forward(self, x):
    self.input = x
    return  corr2d(x, self.weights) + self.biases

  def backward(self, output_gradient, LR):
    kernels_gradient = corr2d(self.input.transpose(1,0,2,3), output_gradient.transpose(1,0,2,3), "valid")
    input_gradient   = conv2d(output_gradient, self.weights.transpose(1,0,2,3), "full")
    self.weights -= LR * kernels_gradient.transpose(1,0,2,3)
    self.biases -= LR * output_gradient.sum(axis=0)
    return input_gradient

class MaxPool2d(Layer):
  def __repr__(self): return f"{self.layer_name}{self.K}"
  def __init__(self, input_shape, K):
    K = K if (isinstance(K, tuple) and len(K)==2) else (K, K)
    self.K = K
    self.input_shape = input_shape
    N, C, ZH, ZW = input_shape # Z.shape Input: NCHW Batch, Channels, Height, Width
    KH, KW = self.K  # Kernel Height & Width
    self.output_shape = N, C, ZH//KH, ZW//KW # the artifact of padding the maxpool
    PadBottom, PadRight = ZH%KH, ZW%KW # How many pixels left on the edge
    self._pad = (0,0),(0,0), (0, PadBottom), (0, PadRight) # to recover the original input shape
    self.layer_name = self.__class__.__name__

  def forward(self,Z, K:tuple=(2,2)):
    KH, KW = self.K  # To avoid the use of self. Maybe Bad programming style but ..
    Ns, Cs, Hs, Ws = Z.strides
    N, C, ZH, ZW = Z.shape #NCHW
    Zstrided = as_strided(Z, shape=(N,C,ZH//KH, ZW//KW, KH, KW), strides=(Ns, Cs, Hs*KH, Ws*KW,Hs, Ws))
    Zstrided = Zstrided.reshape(N,C,ZH//KH, ZW//KW, KH*KW) # reshape to flatten windows to be 1D-vector
    self.MxP = np.max(Zstrided, axis=(-1))
    self.Inx = np.argmax(Zstrided, axis=-1)
    return self.MxP

  def backward(self, output_gradient, *a, **kw):
    ZN, ZC, ZH, ZW = self.MxP.shape
    KH, KW = self.K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))
    for n in range(ZN):
        for c in range(ZC):
            for h in range(ZH):
                for w in range(ZW):
                    ind = self.Inx[n, c, h, w]  # index of max value in pooling window
                    row, col = np.unravel_index(ind, (KH, KW))
                    Z[n, c, h*KH+row, w*KW+col] = self.MxP[n, c, h, w]
    Z = np.pad(Z, self._pad)
    return Z


class Dropout(Layer):
  def __init__(self, input_shape, p):
    self.input_shape, self.output_shape = input_shape, input_shape
    self.p = p
    self.mask = None # New mask must be created for each call.
    self.layer_name = self.__class__.__name__
  def forward(self, x):
    self.mask = np.random.randn(*self.input_shape) < self.p
    x[self.mask] = 0
    return self.x
  def backward(self, output_gradient, *a, **kw):
    input_gradient = np.copy(output_gradient)
    input_gradient[self.mask] = 0
    return input_gradient

class Reshape(Layer):
  def __init__(self, input_shape, output_shape):
    self.input_shape  = input_shape
    self.output_shape = output_shape
    self.layer_name  = self.__class__.__name__
  def forward(self, x): return x.reshape(self.input_shape[0], self.output_shape)
  def backward(self, output_gradient, *a, **kw): return output_gradient.reshape(*self.input_shape)

class Flatten(Layer):
  """Special case of the Reshape where the output is one-dim vector"""
  def __init__(self, input_shape):
    self.input_shape = input_shape
    self.output_shape = input_shape[0], np.prod(input_shape[1:])
    self.layer_name = self.__class__.__name__
  def forward(self, x): return x.reshape(*self.output_shape)
  def backward(self, output_gradient, *a, **kw): return output_gradient.reshape(*self.input_shape)




def sigmoid(x): return 1.0 / (1 + np.exp(-x))
def backward_sigmoid(top_grad, inp_sigmoid): return top_grad * inp_sigmoid * (1 - inp_sigmoid)
def crossentropy(x, y): return np.mean(-np.log(x[np.arange(x.shape[0]), y]))
def softmax(x): temp = np.exp(x - x.max(axis=1, keepdims=True));  return temp / temp.sum(axis=1, keepdims=True)
def softmax_crossentropy(x, y):
    s = softmax(x)
    return crossentropy(s, y), s
def backward_softmax_crossentropy(top_grad, inp_softmax, y):
    res = inp_softmax
    res[np.arange(res.shape[0]), y] -= 1
    return top_grad * res / inp_softmax.shape[0]



class SoftmaxCELayer(Layer):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape

    def forward(self, input_, truth):
        self.input = input_
        self.truth = truth
        self.output, self.cache = softmax_crossentropy(input_, self.truth)
        return self.output

    def backward(self, top_grad=1.0):
        self.bottom_grad = backward_softmax_crossentropy(top_grad, self.cache, self.truth)
        return self.bottom_grad

if __name__ == '__main__':
  lr = 0.1
  x = np.random.randn(2, 10)
  D1 = Dense(x.shape, 5)
  pred = D1(x)
  D1.backward(pred, lr)

  """
  C1 = Conv2d(x.shape, 8, 3)
  # print(C1.output_shape)
  MP1 = MaxPool2d(C1.output_shape, 2)
  # print(MP1.output_shape)
  C2 = Conv2d(MP1.output_shape, 20, 3)
  # print(C2.output_shape)
  MP2 = MaxPool2d(C2.output_shape, 2)
  # print(MP2.output_shape)
  R1 = Flatten(MP2.output_shape)
  # print(R1.output_shape)

  D1 = Dense(R1.output_shape, 100)
  D2 = Dense(D1.output_shape, 10)


  # C1 = Conv2d(x.shape, 5, (3,3))
  # MP1 = MaxPool2d(C1.output_shape, 2)
  # C2 = Conv2d(MP1.output_shape, 8, 4)
  # MP2 = MaxPool2d(C2.output_shape, 2)
  # R1 = Flatten(MP2.output_shape, )
  # print(R1.output_shape)
  # D1 = Dense(R1.output_shape, 100)
  # raise SystemExit
  # D2 = Dense(D1.output_shape, 10)

  x1 = C1(x)
  # print(x1.shape)
  x2 = MP1(x1)
  # print(x2.shape)
  x3 = C2(x2)
  # print(x3.shape)
  x4 = MP2(x3)
  # print(x4.shape)
  x5 = R1(x4)
  # print(x5.shape)
  x6 = D1(x5)
  # print(x6.shape)
  x7 = D2(x6)
  # print(x7.shape)

  print("Forward")
  print(f"Input:     {x.shape}")
  print(f"Conv2d:    {x1.shape}")
  print(f"MaxPool2d: {x2.shape}")
  print(f"Conv2d:    {x3.shape}")
  print(f"MaxPool2d: {x4.shape}")
  print(f"Flatten:   {x5.shape}")
  print(f"Dense:     {x6.shape}")
  print(f"Dense:     {x7.shape}")

  xOutBack = np.random.randn(*x7.shape)
  D2Back = D2.backward(xOutBack, lr)
  D1Back = D1.backward(D2Back, lr)
  R1Back = R1.backward(D1Back)
  MP2Back = MP2.backward(R1Back)
  C2Back = C2.backward(MP2Back, lr)
  MP1Back = MP1.backward(C2Back)
  C1Back = C1.backward(MP1Back, lr)

  print("Backward")
  print(xOutBack.shape)
  print(D2Back.shape)
  print(D1Back.shape)
  print(R1Back.shape)
  print(MP2Back.shape)
  print(C2Back.shape)
  print(MP1Back.shape)
  print(C1Back.shape)
  """


