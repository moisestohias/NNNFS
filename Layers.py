import numpy as np
from math import floor, ceil
as_strided = np.lib.stride_tricks.as_strided
import itertools
from functional import *
from utils.utils import Batcher, MNIST
from activation import Relu

"""
# TODO:

>Note: buttom grad, is what mich refere to as the error, which what propagate backward through the net
Naming convention: inShape, outShape, D:dilation, P: padding, S:stride, G:groups, Z:input, W:weight, B:bias, inF, outF
Maybe we need to go back to more informative naming convention.


Old convP: try this:
wGrad = corr2d(self.input.transpose(1,0,2,3), output_gradient.transpose(1,0,2,3), "valid")
zGrad   = conv2d(output_gradient, self.weights.transpose(1,0,2,3), "full")

"""


class Layer:
    """ All layers should Only acccept batch of inputs: NCHW"""
    def __init__(self): self.trainable = True
    def __call__(self, x): return self.forward(x)
    def __repr__(self): return f"{self.layers_name}(Z)"
    def forward(self, input): raise NotImplementedError
    def backward(self, TopGrad): raise NotImplementedError

class Linear(Layer):
    def __init__(self, inF, outF, bias=True):
        self.layers_name = self.__class__.__name__
        self.trainable = True
        lim = 1 / np.sqrt(inF) # Only inF used to calculate the limit, avoid saturation..
        self.w  = np.random.uniform(-lim, lim, (outF, inF)) # torch style
        self.b = np.random.randn(outF) * 0.1 if bias else None
        self.params = (self.w, self.b)
        self.inShape, self.outShape = (inF,), (outF,)

    def forward(self, z):
        self.z = z
        return affTrans(self.z, self.w, self.b) # [MBS,inF][outF,inF].T -> [MBS,outF]

    def backward(self, TopGrad):
        self.zGrad, self.wGrad, self.bGrad = affTransP(TopGrad, self.z, self.w)
        return self.zGrad


class Conv2d(Layer):
    def __init__(self, inCh, outCh, KS, S=1, P=0, D=1, G=1, b=True, inShape=None):
        if isinstance(KS, int): KS = (KS, KS)
        if isinstance(S, int): self.S = (S, S)
        if isinstance(P, int): self.P = (P, P)
        if isinstance(D, int): self.D = (D, D)
        if inShape: 
            Hin, Win = inShape
            Hout = floor(((Hin+2*self.P[0]-self.D[0]*(KS[0]-1)-1)/self.S[0]) +1) 
            Wout = floor(((Win+2*self.P[1]-self.D[1]*(KS[1]-1)-1)/self.S[1]) +1) 
            self.outShape = outCh, Hout, Wout # we don't need outCh delete that

        self.layers_name = self.__class__.__name__
        self.trainable = True
        self.w = np.random.randn(outCh, inCh, *KS) # (outCh,inCh,H,W)
        self.b = np.random.randn(outCh) if b else None # Each filter has bias, not each conv window
        self.params = [self.w, self.b]
        self.grads = [self.w, self.b]

    def forward(self, x):
        self.z = x
        if self.b is not None: return corr2d(x, self.w) + self.b[np.newaxis, :, np.newaxis, np.newaxis]
        else: return corr2d(x, self.w) 

    def backward(self, TopGrad):
        wGrad, zGrad = corr2d_backward(self.z, self.w, TopGrad)
        if self.b is not None: self.grads = (wGrad, TopGrad.sum(axis=(0, 2, 3))) # wGrad, bGrad
        else: self.grads = (wGrad,) # wGrad
        return zGrad

class Conv1d(Layer):
    """ it still needs some work"""
    def __init__(self, inCh, outCh, KS, S=1, P=0, D=1, G=1, b=True, inShape=None):
        if inShape: self.outShape = outCh, floor(((inShape+2*self.P-self.D*(KS-1)-1)/self.S) +1) # we don't need outCh delete that
        self.layers_name = self.__class__.__name__
        self.trainable = True
        self.w = np.random.randn(outCh, inCh, KS)
        self.b = np.random.randn(outCh) if b else None # Each filter has bias, not each conv window
        self.params = [self.w, self.b]
        self.grads = [self.w, self.b]

    def forward(self, x):
        self.z = x
        if self.b is not None: return corr1d(x, self.w) + self.b[np.newaxis, :, np.newaxis, np.newaxis]
        else: return corr1d(x, self.w) 

    def backward(self, TopGrad):
        wGrad, zGrad = corr1d_backward(self.z, self.w, TopGrad)
        if self.b is not None: self.grads = (wGrad, TopGrad.sum(axis=(0, 2, 3))) # wGrad, bGrad
        else: self.grads = (wGrad,) # wGrad
        return zGrad

class BatchNorm(Layer):
    """TODO: check both forward/backward(gradient) """
    def __init__(self, dim, num_features, eps=1e-05, momentum=0.1):
        self.layers_name = self.__class__.__name__
        self.trainable = True
        self.training = True # Training phase 
        self.momentum = momentum
        self.eps = eps
        self.gamma, self.beta = np.ones(dim), np.zeros(dim) # aka BatchNorm_gain, BatchNorm_bias
        self.runMean, self.runVar = np.zeros(dim), np.ones(dim) # Both runMean/runVar are used for infrence only
        self.params = (self.gamma, self.beta)

    def forward(self, z):
        if not self.training: return self.gamma*(z-self.runMean)/self.runVar + self.beta
        self.z = z
        self.batchMean, self.batchVar = z.mean(0, keepdims=True), z.var(0, keepdims=True)
        self.runMean = self.runMean*(1-self.momentum) + self.batchMean*self.momentum
        self.runVar = self.runVar*(1-self.momentum) + self.batchVar*self.momentum
        zhat = (z-self.batchMean)/np.sqrt(self.batchVar+self.eps)
        return self.gamma*zhat+self.beta

    def backward(self, TopGrad):
        N = self.z.shape[0]
        dZhat = TopGrad * self.gamma
        dsqrtvar = np.sum(dZhat * (self.z - self.batchMean), axis=0) / (self.batchVar + self.eps) # maybe it's np.sqrt(self.batchVar + self.eps)
        zGrad = self.gamma/N *(self.batchVar + self.eps)**(-0.5) * (N * dZhat - np.sum(dZhat, axis=0) - (self.z - self.batchMean) * dsqrtvar)
        self.gamma_grad = np.sum(TopGrad * self.zhat, axis=0)
        self.beta_grad = np.sum(TopGrad, axis=0)   
        return zGrad, (self.gamma_grad, self.beta_grad)


class Reshape(Layer):
    def __init__(self, inShape, outShape):
        self.layers_name = self.__class__.__name__
        self.inShape = inShape if isinstance(inShape,tuple) else (inShape,)
        self.outShape = outShape if isinstance(outShape,tuple) else (outShape,)
    def forward(self, Z): return Z.reshape(Z.shape[0],*self.outShape)
    def backward(self, TopGrad): return np.reshape(TopGrad, self.inShape) # maybe we need TopGrad[1:]

class Flatten(Reshape):
    def __init__(self, inShape):
        self.inShape = inShape if isinstance(inShape,tuple) else (inShape,)
        self.outShape = (np.prod(inShape),)
        self.layers_name = self.__class__.__name__

class MaxPool2d(Layer):
    """ Only batch input is supported NCHW"""
    def __init__(self, KS:tuple=(2,2), S=None, P=0, D=1, return_indices=False, ceil_mode=False, inShape=None):
        self.layers_name = self.__class__.__name__
        if isinstance(KS, int): self.KS = (KS, KS)
        if inShape: self.outShape = calculateConvOutShape(inShape, KS=KS, S=S, P, D=D)

    def forward(self,Z):
        MxP, Inx = maxpool2d(Z, KS)
        self.Inx = Inx
        return MxP
    def backward(self, ZP): #TopGrad
        return unmaxpool2d(ZP, self.Inx)

class Dropout(Layer):
    def __init__(self, inShape, p=0.1):
        self.p = p # Probability to Drop
        self.inShape = inShape
        self.outShape = inShape
        self.layers_name = self.__class__.__name__

    def forward(self, input):
        self.mask = np.random.rand(*self.inShape) < self.p
        output = np.copy(input)
        output[self.mask] = 0
        return output

    def backward(self, TopGrad):
        input_gradient = np.ones(self.inShape)
        input_gradient[self.mask] = 0
        return input_gradient

class LSTM(Layer):
    def __init__(self, input_size, hidden_size, bias=True):
        self.inShape = inShape
        self.outShape = inShape
        self.layers_name = self.__class__.__name__
        self.weight_hh = np.random.randn(hidden_size, 4*hidden_size) # maps previous hidden to new hidden:  torch: .T 
        self.weight_ih = np.random.randn(input_size, 4*hidden_size) # maps input to hidden:  torch: .T
        self.bias      = np.random.randn(hidden_size)
        self.cell_st   = np.random.randn(hidden_size)
        self.hidden_st = np.random.randn(hidden_size)
        self.params    = (self.weight_hh, self.weight_ih, self.bias)

    def forward(self, input):
        self.input = input
        H, c, self.cach = lstm(input, self.weight_hh, self.weight_ih, self.bias, self.cell_st, self.hidden_st)
        return c

    def backward(self, TopGrad):
        zGrad, self.wGrad = lstmP(TopGrad, self.input, self.cach)
        return zGrad

class SoftmaxCELayer(Layer):
    def __init__(self, inShape=None):
        Layer.__init__(self)
        self.layers_name = self.__class__.__name__
        if inShape: self.inShape = inShape if isinstance(inShape,tuple) else (inShape,)

    def forward(self, z, truth):
        self.truth = truth
        self.output, self.cache = softmax_crossentropy(z, self.truth)
        return self.output

    def backward(self, top_grad=1.0):
        self.bottom_grad = backward_softmax_crossentropy(top_grad, self.cache, self.truth)
        return self.bottom_grad

