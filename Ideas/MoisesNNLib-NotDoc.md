"""
+ To be added:
    + all layer must define a method or attribute to return output shape
    + Layers:
        + trainable_parameters, input_shape, output_shape attributes must be stored in the base class Layers
        +
    + MaxPool2d:
    + Conv2d:
        + Stride & Dilation to the Conv2d
        + Non-symmetric kernel_size

+ def __call__ vs __call__ :
    + def __call__(self, input): return self.forward(input) # This doesn't requires forward before
    + __call__ = forward # this line must come after the definition of forward method

+ (N,C,H,W)
    N: Input Samples (aka batch)
    C: Channels
    H: Height
    W: Width

+ The input shape (not only the number of channels but width & height as well) is needed  to be able to calculate the bias shape, which is the same as the output shape.
+ The name 'input' is used rather than 'x' in the forward pass, because during the backward pass we need to refer to it.
"""

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

