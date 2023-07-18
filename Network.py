# Network.py

class Network:
    def __init__(self, optimizer, loss, layers=None): 
        self.layers = if not layers else []
        self.optimizer = optimizer
        self.loss = loss
    def add(self, layer): pass # append a layer to the net.
    def summary(self): pass # Return a summary description of the model
    def fit(self, xtrain=None, ytrain=None, xeval=None, yeval=None, epochs=50, batch_size=32): pass
    def predict(self, x=None): pass


def set_layers_in_out_shape(network):
    """set input_shape & output_shape"""
    for i, layer in enumerate(network.layers):
        if not layer.input_shape: 
            layer.input_shape = network[i - 1].output_shape
        layer.on_input_shape()
        if not layer.output_shape: layer.output_shape = layer.input_shape

def create_model(network, initializer, OptimizerBaseClass, optimizerArgs={}):
    # set input_shape & output_shape
    for i, layer in enumerate(network):
        if not layer.input_shape: 
            layer.input_shape = network[i - 1].output_shape
        layer.on_input_shape()
        if not layer.output_shape: layer.output_shape = layer.input_shape

    # initialize layers & create one optimizer per layer
    layer_shapes = [(layer.input_shape, layer.output_shape) for layer in network]
    initializer.set_layer_shapes(layer_shapes)
    optimizers = []
    for i, layer in enumerate(network):
        initializer.set_layer_index(i)
        param_shapes = layer.initialize(initializer)
        optimizers.append(Optimizer(OptimizerBaseClass, optimizerArgs, param_shapes) if layer.trainable else None)

    # return list of (layer, optimizer)
    return list(zip(network, optimizers))

def summary(model):
    for layer, _ in model:
        print(layer.input_shape, '\t', layer.output_shape)

def forward(model, input):
    output = input
    for layer, _ in model:
        output = layer.forward(output)
    return output

def backward(model, output):
    error = output
    for layer, optimizer in reversed(model):
        error, gradients = layer.backward(error)
        if layer.trainable:
            optimizer.set_gradients(gradients)
    return error

def update(model, iteration):
    for layer, optimizer in model:
        if layer.trainable:
            layer.update(optimizer.get_gradients(iteration))

def train(model, loss, x_train, y_train, epochs, batch=1):
    train_set_size = len(x_train)
    for epoch in range(1, epochs + 1):
        error = 0
        for x, y in zip(x_train, y_train):
            output = forward(model, x)
            error += loss.call(y, output)
            backward(model, loss.prime(y, output))
            if epoch % batch == 0:
                update(model, epoch)
        error /= train_set_size
        print('%d/%d, error=%f' % (epoch, epochs, error))

def test(model, loss, x_test, y_test):
    error = 0
    for x, y in zip(x_test, y_test):
        output = forward(model, x)
        error += loss.call(y, output)
    error /= len(x_test)
    return error
