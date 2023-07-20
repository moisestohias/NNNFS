# ActivationsReWrite.py
import numpy as np

__all__ = [
"Activation",
"Sigmoid",
"Relu",
"Relu6",
"LeakyRelu",
"Elu",
"Swish",
"Tanh",
"Gelu",
"QuickGelu",
"Hardswish",
"Softplus",
"Softmax",
]

class Layer:
  def __init__(self): self.layers_name = self.__class__.__name__ # try remove layer_name here & in childs
  def __call__(self, x): return self.forward(x)
  def __repr__(self): return self.__class__.__name__
  def forward(self, x): raise NotImplementedError
  def backward(self, topGrad, LR): raise NotImplementedError

class Activation(Layer):
  def __init__(self, activation, activationP):
    self.activation = activation
    self.activationP = activationP

  def forward(self, input):
    self.input = input # save input for the backward pass
    return self.activation(self.input)

  def backward(self, topGrad): return np.multiply(topGrad, self.activationP(self.input))


# ReLU, LeakyReLU, ELU enable faster and better convergence than sigmoids.
# GELU: Gaussian Error Linear Unit used in most Transformers(GPT-3, BERT): paperswithcode.com/method/gelu
# Hard-Swish: paperswithcode.com/method/hard-swish

## TODO: make sure you are Clipping correctly

def sigmoid(x): return np.reciprocal((1.0+np.exp(-np.clip(x, -100, 2e12))))
def sigmoidP(x):
  s = 1.0/( (1.0+np.exp(-np.clip(x, -100, 2e12))) )
  return s * (1 - s) # σ(x)*(1-σ(x))
def relu(x): return np.where(x>= 0, x, 0)
def reluP(x): return np.where(x>= 0, 1, 0)
def leaky_relu(x, alpha=0.01): return np.where(x>= 0, x, alpha*x)
def leaky_reluP(x, alpha=0.01): return np.where(x>= 0, 1, alpha)
def elu(x, alpha=0.01): return np.where(x>= 0, x, alpha*(np.exp(np.clip(x, -1e15, 1e15))-1))
def eluP(x, alpha=0.01): return np.where(x>= 0, 1, alpha*np.exp(np.clip(x, -1e15, 1e15)))
def swish(x): return x * np.reciprocal((1.0+np.exp(-np.clip(x, -1e15, 1e15)))) # x*σ(x) σ(x)+σ'(x)x : σ(x)+σ(x)*(1-σ(x))*x
def swishP(x): s = np.reciprocal((1.0+np.exp(-np.clip(x, -1e15, 1e15)))); return s+s*(1-s)*x #σ(x)+σ(x)*(1-σ(x))*x
silu, siluP = swish, swishP # The SiLU function is also known as the swish function.
def tanh(x): return np.tanh(x) # or 2.0*(σ((2.0 * x)))-1.0
def tanhP(x): return 1 - np.tanh(x) ** 2
def gelu(x): return 0.5*x*(1+np.tanh(0.7978845608*(x+0.044715*np.power(x,3)))) # sqrt(2/pi)=0.7978845608
def geluP(x): return NotImplemented#Yet Error
def quick_gelu(x): return x*sigmoid(x*1.702) # faster version but inacurate
def quick_geluP(x): return 1.702*sigmoidP(x*1.702)
def hardswish(x): return x*relu(x+3.0)/6.0
def hardswishP(x): return 1.0/6.0 *relu(x+3)*(x+1.0)
def softplus(x, limit=20.0, beta=1.0): return (1.0/beta) * np.log(1 + np.exp(x*beta))
def softplusP(limit=20, beta=1.0): _s = np.exp(x*beta) ; return (beta*_s)/(1+_s)
def relu6(x): return relu(x)-relu(x-6)
def relu6P(x): return reluP(x)-reluP(x-6)

class Sigmoid(Activation):
  def __init__(self): super().__init__(sigmoid, sigmoidP)
class Relu(Activation):
  def __init__(self): super().__init__(relu, reluP)
class Relu6(Activation):
  def __init__(self): super().__init__(relu6, relu6P)
class LeakyRelu(Activation):
  def __init__(self, alpha=0.01): super().__init__(leaky_relu, leaky_reluP)
class Elu(Activation):
  def __init__(self, alpha=0.01): super().__init__(elu, eluP)
class Swish(Activation):
  def __init__(self): super().__init__(swish, swishP)
class Tanh(Activation):
  def __init__(self): super().__init__(tanh, tanhP)
class Gelu(Activation):
  def __init__(self): super().__init__(gelu, geluP)
class QuickGelu(Activation):
  def __init__(self): super().__init__(quick_gelu, quick_geluP)
class Hardswish(Activation):
  def __init__(self): super().__init__(hardswish, hardswishP)
class Softplus(Activation):
  def __init__(self, limit=20.0, beta=1.0): super().__init__(softplus, softplusP)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    def backward(self, topGrad):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, topGrad)
"""
# Plot
import matplotlib.pyplot as plt
x = np.arange(-6, 7, 0.1)
plt.title("ReLU and Swish functions", fontsize = 16)
plt.plot(sigmoid(x), label="Sigmoid(x)")
plt.legend(prop={'size': 10})
plt.grid()
plt.axes()
plt.show()
"""
