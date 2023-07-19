import numpy as np
import math
# from scipy import signal


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




