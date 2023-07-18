from utils.utils import MNIST, Batcher, one_hot_batch
import itertools
import pickle, gzip
import tqdm
from tqdm import trange
import sys
import numpy as np


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

def flatten(x): return x.reshape((x.shape[0], -1)), x.shape
def affine_transform(input_, weight, bias): return input_.dot(weight) + bias
def backward_affine_transform(top_grad, input_, weight):
    bias_grad = top_grad.sum(axis=0)
    weight_grad = input_.T.dot(top_grad)
    input_grad = top_grad.dot(weight.T)
    return input_grad, weight_grad, bias_grad

class Layer:
    def __init__(self):
        self.built = False
        self.input_shape = self.output_shape = None
        self.params = []
        self.grads = []

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
        self.output = input_.dot(self.weights) + self.bias
        return self.output

    def backward(self, top_grad):
        bias_grad = top_grad.sum(axis=0)
        weight_grad = self.input.T.dot(top_grad)
        input_grad = top_grad.dot(self.weights.T)
        self.grads[0][...] = weight_grad
        self.grads[1][...] = bias_grad
        self.bottom_grad = input_grad
        return self.bottom_grad


class SigmoidLayer(Layer):
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
        self.output = sigmoid(input_)
        return self.output

    def backward(self, top_grad):
        self.bottom_grad = backward_sigmoid(top_grad, self.output)
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
        self.output, self.cache =  input_.reshape((input_.shape[0], -1)), input_.shape
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
        """
        :param input_: The logits
        :param truth: The indices of the correct classification
        :return: The calculated loss
        """
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
        """
        Run the network for k layers.
        :param k: If positive, run for the first k layers, if negative, ignore the last -k layers. Cannot be 0.
        :param input_: The input to the network
        :return: The output of the second last layer
        """
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


# ================================================================
def load_data():
    (trainx, trainy), (valx, valy), (testx, testy) = pickle.load(gzip.open(r"D:\DS\NNDS\mnist_one_hot.pkl.gz"),
                                                                 encoding="latin1")
    trainy = np.argmax(trainy, axis=1)
    valy = np.argmax(valy, axis=1)
    testy = np.argmax(testy, axis=1)
    trainx = trainx * 2 - 1
    valx = valx * 2 - 1
    testx = testx * 2 - 1
    return (trainx.reshape(-1, 28*28), trainy), (valx.reshape(-1, 28*28), valy), (testx.reshape(-1, 28*28), testy)

def train_classifier(data, n_iters=1, batch_size=100):
    print(f'Training NN classifier for {n_iters} iterations.')
    (trainx, trainy), (valx, valy), (testx, testy) = data
    train_size, val_size, test_size = trainx.shape[0], valx.shape[0], testx.shape[0]
    train_batches = (train_size - 1) // batch_size + 1
    val_batches = (val_size - 1) // batch_size + 1
    test_batches = (test_size - 1) // batch_size + 1

    model = Network()
    model.add_layer(FCLayer(100)) \
        .add_layer(SigmoidLayer()) \
        .add_layer(FCLayer(10)) \
        .add_layer(SoftmaxCELayer())
    for i in range(1, n_iters + 1):
        train_order = np.random.permutation(train_size)

        for j in range(train_batches):
            cost = model.forward(trainx[train_order[j * batch_size: (j + 1) * batch_size]],
                                 trainy[train_order[j * batch_size: (j + 1) * batch_size]])
            print(f'Curr loss: {cost}')
            model.backward()
            model.adam_trainstep()
        correct = []
        for j in range(val_batches):
            res = model.run(valx[j * batch_size:(j + 1) * batch_size])
            correct.append(np.argmax(res, axis=1) == valy[j * batch_size:(j + 1) * batch_size])
        print(f'Validation accuracy: {np.mean(correct)}')
        print('-------------------------------------------------------')

    correct = []
    for i in range(test_batches):
        res = model.run(testx[i * batch_size:(i + 1) * batch_size])
        correct.append(np.argmax(res, axis=1) == testy[i * batch_size:(i + 1) * batch_size])
    print(f'Test accuracy: {np.mean(correct)}')
    print('-------------------------------------------------------')


data = load_data()
train_classifier(data)
