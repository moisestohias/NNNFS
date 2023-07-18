# MoisesNNLib-Doc.py

"""
Design philosophy:
Activation function also are treated as layer,
Each layer, must define two building block methods:
    forward : Take an input return output
    backward: Take the Accumulated Gradient (AG) from next layer, calculate the gradient w.r.t its weight and biases, update its parameters, calculate and return its input gradient (aka error). 



Calculate the size of the output layer:
kvirajdatt.medium.com/calculating-output-dimensions-in-a-cnn-for-convolution-and-pooling-layers-with-keras-682960c73870
    + I (ixi)    : The input dimensions of the image
    + k (kxk)    : The size of filter/kernel
    + S (integer): Strides
    + P (integer): Padding
    + D (integer): Depth/Number of feature maps/activation maps
Conv = [(I - K +2 *P) / S] +1 x D
Pool = [(I - K) / S] + 1 x D
"""


class Layer:
    def __i/nit__(self): raise NotImplementedError
    def forward(self, x): raise NotImplementedError
    def backward(self, x): raise NotImplementedError

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.activation_prime(self.input)

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x : 1. / 1. + np.exp(-x)
        sigmoid_prime = lambda x : 1. 
        super().__init__(sigmoid, sigmoid_prime)

class Relu(Layer):
    def __init__(self):
        relu = lambda x : 1. / 1. + np.exp(x)
        relu_prime = lambda x : 1. / 1. + np.exp(x)
        super().__init__(relu, relu_prime)

class LeakyRelu(Layer):
    def __init__(self, alpha=0.02):
        leaykyRealy = lambda x : np.where(x>= 0, x, x*alpha)
        leaykyRealy_prime = lambda x : np.where(x>= 0, 1, alpha)
        super().__init__(leaykyRealy, leaykyRealy_prime)

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Weights: Like Micheal implementation rows represent nodes in the current layer
        lim = 1.0/np.sqrt(input_size)
        self.weights = np.random.uniform(-lim, lim, size=(output_size, input_size)) 
        self.biases = np.random.randn(output_size, 1)
    def foreward(self, input):
        self.input # save the input for the BP pass.
        # Note: this is not the activation of the layer, we still need to apply the activation function -Non linearity-
        return self.weights.dot(self.input) + self.biases
    def backward(self, output_gradient, learning_rate): 
        # First calculate the gradient w.r.t w & b, then return the input gradient (aka error) 
        # The way we calculate the weights_gradient in dense layer, we multiple the output gradient (aka error of the next layer) with the activation (aka input) of the current layer. Where both are vectors, the result must be a matrix whose row are exactly equal to number of neuron on the next layer (where the error came from) -sticking with Mich weight repr-
        # Note: we don't calculate the gradient with respect of b, because it's equal to the error of the next layer. Or more specifically error * 1 
        weights_gradient = self.input.T.dot(output_gradient)
        self.weights = self.weights - (learning_rate * weights_gradient)
        self.biases = self.biases - (learning_rate * output_gradient)

        # Note: This is not the full error yet, we still need to multiply by the derivative of the Activation Function.
        return self.weights.T.dot(output_gradient)
        # This is one of the design aspect that we need to keep in mind. How we want to represent ActFun either as independent layer of a component of the layer.


class NeuralNetwork:
    def predict(self, x): raise NotImplementedError
    def Back(self, x): raise NotImplementedError
    def SGD(self, training_data, test_data=None, epochs=10, batch_size=32, learning_rate=0.3):
    n = len(training_data)
    batches = [training_data[k, k+mini_batch_size] for k in range(0, n, batch_size)]
    


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

        for i in range(self.depth):
            for j in range(self.channels):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        # Here: Define a function responsible for updating the params be be able to freeze layers.
        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
