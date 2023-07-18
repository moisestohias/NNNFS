import numpy as np
import itertools
from functional import *
from activation import Relu
from utils.utils import Batcher, MNIST
as_strided = np.lib.stride_tricks.as_strided
mag = np.linalg.norm

class Layer:
    def __init__(self):
        self.built = False
        self.input_shape = self.output_shape = None
        self.params = []
        self.grads = []
    def __call__(self, x): return self.forward(x)


class ConvLayer(Layer):
    def __init__(self, n_filters, filter_shape, stride=(1, 1), dilation=1):
        Layer.__init__(self)
        self.filter_shape = filter_shape
        self.stride = stride
        self.dilation = dilation
        self.n_filters = n_filters

    def build(self, input_shape):
        self.input_shape = input_shape
        fan_in = input_shape[0] * self.filter_shape[0] * self.filter_shape[1]
        fan_out = self.n_filters * self.filter_shape[0] * self.filter_shape[1]
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        self.filters = np.random.normal(0.0, stddev,
                                        size=(self.n_filters, self.input_shape[0],
                                              self.filter_shape[0], self.filter_shape[1]))
        self.bias = np.ones((self.n_filters,)) * 0.01
        self.params = [self.filters, self.bias]
        dilated_shape = ((self.filter_shape[0] - 1) * self.dilation + 1, (self.filter_shape[1] - 1) * self.dilation + 1)
        self.output_shape = (self.n_filters,
                             (input_shape[1] - dilated_shape[0]) // self.stride[0] + 1,
                             (input_shape[2] - dilated_shape[1]) // self.stride[1] + 1)
        self.grads = [np.empty_like(param) for param in self.params]
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output = corr2d(input_, self.filters) + self.bias[np.newaxis, :, np.newaxis, np.newaxis]
        return self.output

    def backward(self, top_grad):
        self.bottom_grad, self.grads[0][...] = backward_conv2d(top_grad, self.input, self.filters, self.dilation, self.stride)
        self.grads[1][...] = top_grad.sum(axis=(0, 2, 3))
        return self.bottom_grad


class FCLayer(Layer):
    def __init__(self, n_units):
        Layer.__init__(self)
        self.n_units = n_units

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (self.n_units,)
        stddev = np.sqrt(2.0 / (self.input_shape[0] + self.n_units))
        self.weights = np.random.normal(0.0, stddev, size=(self.input_shape[0], self.n_units))
        self.bias = np.ones((self.n_units,)) * 0.01
        self.params = [self.weights, self.bias]
        self.grads = [np.empty_like(param) for param in self.params]
        self.built = True
    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output = affine_transform(input_, self.weights, self.bias)
        return self.output

    def backward(self, top_grad):
        input_grad, weight_grad, bias_grad = backward_affine_transform(top_grad, self.input, self.weights)
        self.grads[0][...] = weight_grad
        self.grads[1][...] = bias_grad
        self.bottom_grad = input_grad
        return self.bottom_grad



class FlattenLayer(Layer):
    def __init__(self): Layer.__init__(self)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output, self.cache = flatten(input_)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_flatten(top_grad, self.cache)
        return self.bottom_grad

class SSELayer(Layer):
    def __init__(self): Layer.__init__(self)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = ()
        self.built = True

    def forward(self, input_, truth):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.truth = truth
        self.output =  ((input_ - self.truth) ** 2).sum() / input_.shape[0]
        return self.output

    def backward(self, top_grad=1.0):
        self.bottom_grad = top_grad * 2 * (self.input - self.truth) / self.input.shape[0]
        return self.bottom_grad


class Network:
    def __init__(self):
        self.layers = []
        self.params = []
        self.grads = []
        self.optimizer_built = False

    def add_layer(self, layer):
        self.layers.append(layer)
        return self

    def forward(self, input_, truth):
        input_ = self.run(input_)
        return self.layers[-1].forward(input_, truth)

    def run(self, input_, k=-1):
        k = len(self.layers) if not k else k
        for layer in self.layers[:min(len(self.layers) - 1, k)]:
            input_ = layer.forward(input_)
        return input_

    def backward(self):
        top_grad = 1.0
        for layer in self.layers[::-1]:
            top_grad = layer.backward(top_grad)

    def adam_trainstep(self, alpha=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8, l2=0.):
        if not self.optimizer_built:
            self.params.extend(itertools.chain(*[layer.params for layer in self.layers]))
            self.grads.extend(itertools.chain(*[layer.grads for layer in self.layers]))
            self.first_moments = [np.zeros_like(param) for param in self.params]
            self.second_moments = [np.zeros_like(param) for param in self.params]
            self.time_step = 1
            self.optimizer_built = True
        for param, grad, first_moment, second_moment in zip(self.params, self.grads,
                                                            self.first_moments, self.second_moments):
            first_moment *= beta_1
            first_moment += (1 - beta_1) * grad
            second_moment *= beta_2
            second_moment += (1 - beta_2) * (grad ** 2)
            m_hat = first_moment / (1 - beta_1 ** self.time_step)
            v_hat = second_moment / (1 - beta_2 ** self.time_step)
            param -= alpha * m_hat / (np.sqrt(v_hat) + epsilon) + l2 * param
        self.time_step += 1


class ReluLayer(Layer):
    def __init__(self): Layer.__init__(self)
    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.built = True

    def forward(self, input_):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.output, self.cache = relu(input_)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_relu(top_grad, self.cache)
        return self.bottom_grad

class SoftmaxCELayer(Layer):
    def __init__(self): Layer.__init__(self)

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = ()
        self.built = True

    def forward(self, input_, truth):
        if not self.built:
            input_shape = input_.shape[1:]
            self.build(input_shape)
        self.input = input_
        self.truth = truth
        self.output, self.cache = softmax_crossentropy(input_, self.truth)
        return self.output

    def backward(self, top_grad=1.0):
        self.bottom_grad = backward_softmax_crossentropy(top_grad, self.cache, self.truth)
        return self.bottom_grad

def crossentropy(x, y): return np.mean(-np.log(x[np.arange(x.shape[0]), y]))
def backward_crossentropy(top_grad, x, y):
    res = np.zeros(x.shape, dtype=x.dtype)
    res[np.arange(x.shape[0]), y] = - np.reciprocal(x[np.arange(x.shape[0]), y]) / x.shape[0]
    return res * top_grad


def softmax(x):
    temp = np.exp(x - x.max(axis=1, keepdims=True))
    res = temp / temp.sum(axis=1, keepdims=True)
    return res
def softmax_crossentropy(x, y):
    s = softmax(x)
    return crossentropy(s, y), s
def backward_softmax_crossentropy(top_grad, inp_softmax, y):
    res = inp_softmax
    res[np.arange(res.shape[0]), y] -= 1
    return top_grad * res / inp_softmax.shape[0]

def flatten(x): return x.reshape((x.shape[0], -1)), x.shape
def backward_flatten(top_grad, original_shape): return top_grad.reshape(original_shape)
def relu(x):
    cache = x > 0
    return x * cache, cache
def backward_relu(top_grad, cache): return top_grad * cache


# ================================================================
def train_classifier(data, n_iters=3, batch_size=10):
    print(f'Training a dilated CNN classifier for {n_iters} iterations.')
    (trainx, trainy), (valx, valy), (testx, testy) = data
    train_size, val_size, test_size = trainx.shape[0], valx.shape[0], testx.shape[0]
    train_batches = (train_size - 1) // batch_size + 1
    val_batches = (val_size - 1) // batch_size + 1
    test_batches = (test_size - 1) // batch_size + 1

    model = Network()
    model.add_layer(FlattenLayer()) \
        .add_layer(FCLayer(30)) \
        .add_layer(ReluLayer()) \
        .add_layer(FCLayer(10)) \
        .add_layer(SoftmaxCELayer())


    for i in range(1, n_iters + 1):
        train_order = np.random.permutation(train_size)
        for j in range(train_batches):
            cost = model.forward(trainx[train_order[j * batch_size: (j + 1) * batch_size]],
                                 trainy[train_order[j * batch_size: (j + 1) * batch_size]])
            model.backward()
            model.adam_trainstep()
        correct = []
        for j in range(val_batches):
            res = model.run(valx[j * batch_size:(j + 1) * batch_size])
            correct.append(np.argmax(res, axis=1) == valy[j * batch_size:(j + 1) * batch_size])
        print(len(correct))
        print(f'Validation accuracy: {np.mean(correct)}')
        print('-------------------------------------------------------')

    correct = []
    for i in range(test_batches):
        res = model.run(testx[i * batch_size:(i + 1) * batch_size])
        correct.append(np.argmax(res, axis=1) == testy[i * batch_size:(i + 1) * batch_size])
    print(f'Test accuracy: {np.mean(correct)}')
    print('-------------------------------------------------------')

def load_mmist(Standardize=True, Flat=False):
  """Load the MNIST data"""
  import pickle, gzip, sys
  platform = sys.platform
  MNISTPath =  r"E:\DS\NNDS\mnist.pkl.gz" if platform == "windows" else "/home/moises/.DS/NNDS/mnist.pkl.gz"
  (trainx, trainy), (valx, valy), (testx, testy) = pickle.load(gzip.open(MNISTPath), encoding="latin1")
  # ((50000, 784) (50000,)) ((10000, 784) (10000,)) ((10000, 784) (10000,))
  if not Standardize and Flat: return (trainx, trainy), (valx, valy), (testx, testy)
  if Standardize: 
    # You can standardize either per-sample or on the entire dataset: trainx = trainx - trainx.mean() 
    trainx = trainx - trainx.mean(axis=1)[:,None] # Standardizing per-sample
    valx = valx - valx.mean(axis=1)[:,None]
    testx = testx - testx.mean(axis=1)[:,None]
  if Flat: return (trainx, trainy), (valx, valy), (testx, testy)
  return (trainx.reshape(-1,1, 28, 28), trainy), (valx.reshape(-1,1, 28, 28), valy), (testx.reshape(-1,1, 28, 28), testy)

train_classifier(load_mmist() )

