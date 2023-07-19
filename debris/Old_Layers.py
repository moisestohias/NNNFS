"""
This module 
"""
import numpy as np
from scipy import signal
import math


class Conv2d(Layer):
    """
    BackProp in CNN Explanation & How To: youtu.be/Lakz2MoHy6o
    *: Convolution =
    ⋆: Correlation
    !Note: X*K = X⋆rot180(K)
    """

    def __init__(self, input_shape, depth, kernel_size):
        #  stride & dilation to be added
        # Only symmetric kernel_size is allowed for now.

        """
        We need to know the input_shape: channels, height & width once the layer is created,
        because we need to Create & Initialize the layer's weight & biases
            + weights shape is calculated using the kernel and output depth
            + biases shape is calculated using the input shape and kernel size & output depth.
        Input channels is also needed during the forward/backward pass
        Ouput Shape can be calculated from the output of the convolution
        """

        self.input_shape = input_shape
        self.channels, self.input_height, self.input_width = input_shape
        self.depth = depth # Number of filters.
        self.output_shape = (depth, self.input_height - kernel_size + 1, self.input_width - kernel_size + 1)
        self.kernels_shape = (depth, self.channels, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape) # (depth, channels,height, width)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, x):
        self.input = x # Storing the input for the backward pass.
        self.output = np.copy(self.biases) # copy the bias instead zero_like avoiding sum in the loop
        for i in range(self.depth): # loop over depth first, each out channel is independent
            for j in range(self.channels): # loop over in_channel second, all channels must be summed.
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        # Using the output_gradient, we calculate the kernels_gradient, and the input_gradient.
        # kernels_gradient = ∂E/∂K_ij = X_j⋅∂E/Y_i = X⋆∂E/Y_i
        # inpt_gradient = output_gradient*K = output_gradient⋆rot180(K) !Note: both *&⋆ full version
        # biases_gradient = output_gradient
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        # Supporting batching
        for b in range(self.MBS)
            for i in range(self.depth):
                for j in range(self.channels):
                    kernels_gradient[i, j] = signal.correlate2d(self.input[b,j], output_gradient[b,i], "valid")
                    input_gradient[b, j] += signal.convolve2d(output_gradient[b,i], self.kernels[i, j], "full")

        # Here: Define a function responsible for updating the params be be able to freeze layers.
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient



class MaxPool2d(Layer):
    """ Only batch input is supported NCHW"""
    def __init__(self, ZShape, K:tuple=(2,2), MustPad=False):
        self.layers_name = self.__class__.__name__
        self.MustPad = MustPad
        N, C, ZH, ZW = ZShape #NCHW
        if isinstance(K, int): self.K = (K, K)
        KH, KW = self.K  #HW
        self.output_shape = N, C, ZH//KH, ZW//KW
        EdgeH, EdgeW = ZH%KH, ZW%KW # How many pixels left on the edge
        if MustPad and (EdgeH!=0 or EdgeW!=0): # If Pad=True & there are pixelx left we pad
            PadH, PadW = KH-EdgeH,KW-EdgeW
            PadTop, PadBottom = ceil(PadH/2), floor(PadH/2)
            PadLeft, PadRight = ceil(PadW/2), floor(PadW/2)
            self.Padding = ((0,0),(0,0), (PadTop, PadBottom), (PadLeft, PadRight))
    def forward(self,Z):
        Ns, Cs, Hs, Ws = Z.strides
        N, C, ZH, ZW = Z.shape #NCHW
        KH, KW = self.K  #HW
        if self.MustPad:
            Z = np.pad(Z, self.Padding)
            N, C, ZH, ZW = Z.shape #NCHW
            Ns, Cs, Hs, Ws = Z.strides
        return as_strided(Z, shape=(N,C,ZH//KH, ZW//KW, KH, KW), strides=(Ns, Cs, Hs*KH, Ws*KW,Hs, Ws)).max(axis=(-2,-1))
    def backward(self, ZP): #TopGrad
        KH, KW = self.K
        N, C, ZPH, ZPW, = ZP.shape #NCHW
        ZPNs, ZPCs, ZPHs, ZPWs = ZP.strides
        a = as_strided(ZP, shape=(N,C,ZPH, ZPW, KH, KW), strides=(ZPNs,ZPCs,ZPHs,ZPWs,0,0))
        return a.transpose(0,1,2,4,3,5).reshape(N, C, ZPH*KH, ZPW*KW) # final output shape

if __name__ == "__main__":
    lr = 0.001
    x = np.random.randn(10, 3, 28, 28)
    C1 = Conv2d(x.shape, 10, 3)
    print(C1(x).shape)
    """

    C2 = Conv2d(C1.output_shape, 8, 3)
    MP1 = MaxPool2d(C2.output_shape, (2,2))
    ConvDrop = Dropout(MP1.output_shape, 0.5)
    C3 = Conv2d(ConvDrop.output_shape, 6, 3)
    R = Reshape(C3.output_shape, 6*22*22)
    D1 = Dense(R.output_shape, 100)
    DenseDrop = Dropout((100,1), 0.2)
    D2 = Dense(100, 10)

    x1 = C1(x)
    x2 = C2(x1)
    x2Pooled = MP1(x2)
    x2PooledDroped = ConvDrop(x2Pooled)
    x3 = C3(x2PooledDroped)
    x4 = R(x3)
    # x5 = D1(x4)
    # x5 = DenseDrop(x5)
    # x6 = D2(x5)
    """

    # print("# Forward===========")
    # print("Input:",x.shape)
    # print(x1.shape)
    # print(x2.shape)
    # print(x2Pooled.shape)
    # print(x2PooledDroped.shape)
    # print(x3.shape)
    # print(x4.shape)
    # print(x5.shape)
    # print(x6.shape)

    # x_back1 = D2.backward(x6, lr)
    # x_back2 = D1.backward(x_back1, lr)
    # x_back3 = R.backward( x_back2, lr)
    # x_back4 = C3.backward(x_back3, lr)
    # x_back5 = C2.backward(x_back4, lr)
    # x_back6 = C1.backward(x_back5, lr)

    # print("# Backward===========")
    # print("# Dense---")
    # print(x_back1.shape)
    # print(x_back2.shape)
    # print("# Reshape---")
    # print(x_back3.shape)
    # print("# Conv---")
    # print(x_back4.shape)
    # print(x_back5.shape)
    # print(x_back6.shape)
