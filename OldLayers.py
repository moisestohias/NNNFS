import numpy as np
import math
# from scipy import signal


"""
a = z@w
da/dw = z.T # wGrad = z.T.dot(topGrad)
da/dz = w.T # zGrad = np.dot(topGrad, w.T)

"""

class Layer:
    """ All layers should Only acccept batch of zs: NCHW"""
    def __init__(self): self.trainable = True
    def __call__(self, x): return self.forward(x)
    def __repr__(self): return f"{self.layers_name}(Z)"
    def forward(self, z): raise NotImplementedError
    def backward(self, TopGrad): raise NotImplementedError

class Dense(Layer):
    def __init__(self, inF, outF):
        self.inF = inF
        self.outF = outF 
        self.layers_name = self.__class__.__name__
        lim = 1 / math.sqrt(inF)
        self.weight  = np.random.uniform(-lim, lim, (inF, outF))
        self.bias = np.random.randn(outF)

    def forward(self, z):
        self.z = z
        return self.z.dot(self.weight) + self.bias

    def backward(self, topGrad, LR):
        wGrad = self.z.T.dot(topGrad)
        zGrad = np.dot(topGrad, self.weight.T)
        self.weight -= LR * wGrad
        self.bias -= LR * topGrad.sum(0)
        return zGrad


class Conv2d(Layer):
    # in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None
    def __init__(self, z_shape, depth, kernel_size):
        self.ZC, self.ZW, self.ZH = z_shape
        self.depth = depth # Number of filters.
        self.output_shape = (depth, self.ZH - kernel_size + 1, self.ZW - kernel_size + 1)
        self.kernels_shape = (depth, self.ZC, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)

    def __call__(self, z): return self.forward(z)
    def forward(self, z):
        self.z = z
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.ZC):
                self.output[i] = signal.correlate2d(self.z[j], self.kernels[i, j], "valid")
        return self.output, self.output_shape

    def backward(self, topGrad, LR):
        kernels_gradient = np.zeros(self.kernels_shape)
        zGrad = np.zeros(self.z_shape)

        for i in range(self.depth):
            for j in range(self.ZC):
                kernels_gradient[i, j] = signal.correlate2d(self.z[j], topGrad[i], "valid")
                zGrad[j] += signal.convolve2d(topGrad[i], self.kernels[i, j], "full")

        self.kernels -= LR * kernels_gradient
        self.biases -= LR * topGrad
        return zGrad

class Reshape(Layer):
    def __init__(self, z_shape, output_shape):
        self.z_shape = z_shape
        self.output_shape = output_shape
    def forward(self, z): return np.reshape(z, self.output_shape)

    def backward(self, topGrad, LR):
        return np.reshape(topGrad, self.z_shape)

def mse_prime(y, p): return (p-y).mean()
def mse(y, p): return 0.5*np.power((y-p), 2).mean()


def sigmoid(x): return np.reciprocal((1.0+np.exp(-x)))
def sigmoid_prime(x): s = np.reciprocal((1.0+np.exp(-x))); return s * (1 - s) # σ(x)*(1-σ(x))
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def __call__(self, z): return self.forward(z)
    def forward(self, z):
        self.z = z # save z for the backward pass
        return self.activation(self.z)

    def backward(self, topGrad, LR):
        return np.multiply(topGrad, self.activation_prime(self.z))

class Sigmoid(Activation):
  def __init__(self): super().__init__(sigmoid, sigmoid_prime)



def trainSine(lr = 0.01):
    x = np.sin(np.linspace(-10, 10, 100))
    D1 = Dense(x.shape[-1], 100)
    D2 = Dense(D1.outF, 100)
    S1 = Sigmoid()
    for i in range(100):
        x1 = S1(D1.forward(x))
        x2 = S1(D2.forward(x1))
        loss = mse(x, x2)
        print(loss)
        OutGrad = mse_prime(x, x2)
        D2Back = S1.backward(D2.backward(OutGrad, lr), lr)
        D1Back = S1.backward(D1.backward(D2Back, lr), lr)


if __name__ == "__main__":
    # x = np.random.randn(3,5)
    # D1 = Dense(x.shape[-1], 4)
    # x1 = D1.forward(x)
    # OutGrad = np.ones(x1.shape)
    # D1Back =  D1.backward(OutGrad, lr)

    # D2 = Dense(D1.output_size, 100)
    # x1 = D1.forward(x)
    # x2 = D2.forward(x1)
    # OutGrad = np.ones(x2.shape)
    # D2Back =  D2.backward(OutGrad, lr)
    # D1Back =  D1.backward(D2Back, lr)
    # print(OutGrad.shape)
    # print(D2Back.shape)
    # print(D1Back.shape)



