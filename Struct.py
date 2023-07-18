from NewLayers import Dense, Conv2d, MaxPool2d, Flatten, SoftmaxCELayer, np
from activation import Sigmoid
from functional import mse, mse_prime
from utils.utils import MNIST, Batcher, one_hot_batch


class Network:
    def __init__(self, layers=None, LR=0.001):
      self.layers = layers if layers else []
      self.LR = LR
    def __getitem__(self, k): return self.layers[k]
    def __iter__(self): return iter(self.layers)

    def __repr__(self):
      representation = ""
      for l in self.layers:
        if hasattr(l, "input_shape") and hasattr(l, "output_shape"): rep = f"{l}  InputShape: {l.input_shape}  OutputShape: {l.output_shape}\n"
        else: rep = repr(l) + "\n"
        representation += rep
      return representation

    def __call__(self, a):
      for layer in self.layers: a = layer(a)
      return a
    def backward(self, output_gradient):
      for layer in reversed(self.layers): output_gradient = layer.backward(output_gradient, self.LR)
      return output_gradient

    def add(self, layer):  self.layers.append(add)
    def summary(self): pass # Return a summary description of the model
    def fit(self, xtrain=None, ytrain=None, xeval=None, yeval=None, epochs=50, batch_size=32): pass


MBS = 100
LR = 0.1
mnist = Batcher(MNIST(), MBS)
"""
InputShape = (MBS, 28*28)
D1  = Dense(InputShape, 100)
S1 = Sigmoid()
D2  = Dense(D1.output_shape, 10)
S2 = Sigmoid()
model = Network((D1, S1, D2, S2), LR)
SM = SoftmaxCELayer(D2.output_shape)

# DenseNet
for i in range(5):
  print("----------------")
  for x, y in Batcher(MNIST(), MBS):
    # y = one_hot_batch(y)
    pred = model(x.reshape(-1, 784))
    loss = SM(pred, y)
    # print(loss)
    acc = (sum(np.argmax(pred, axis=-1)==y)/MBS) *100
    print(f"Accuracy {acc:.4f}%")
    model.backward(SM.backward())
"""


# # ConvNet
InputShape = (MBS,1,28,28)
C1  = Conv2d(InputShape, 8, 3)
MP1 = MaxPool2d(C1.output_shape, 2)
S1 = Sigmoid()
C2  = Conv2d(MP1.output_shape, 8, 3)
MP2 = MaxPool2d(C2.output_shape, 2)
S2 = Sigmoid()
F1  = Flatten(MP2.output_shape)
D1  = Dense(F1.output_shape, 100)
S3 = Sigmoid()
D2  = Dense(D1.output_shape, 10)
# S4 = Sigmoid()
model = Network((C1, MP1, S1, C2, MP2, S2, F1, D1, S3, D2), LR)
SM = SoftmaxCELayer(D2.output_shape)

for x, y in mnist:
  pred = model(x)
  loss = SM(pred, y)
  print(loss)
  acc = (sum(np.argmax(pred, axis=-1)==y)/MBS) *100
  print(f"Accuracy {acc:.4f}%")
  model.backward(SM.backward())







# InputShape = (MBS,1,28,28)
# C1  = Conv2d(InputShape, 8, 3)
# MP1 = MaxPool2d(C1.output_shape, 2)
# S1 = Sigmoid()
# C2  = Conv2d(MP1.output_shape, 8, 3)
# MP2 = MaxPool2d(C2.output_shape, 2)
# S2 = Sigmoid()
# C2  = Conv2d(MP1.output_shape, 2, 3)
# MP2 = MaxPool2d(C2.output_shape, 2)
# S3 = Sigmoid()
# C2  = Conv2d(MP1.output_shape, 1, 3)
# S4 = Sigmoid()
# model = Network((C1, MP1, S1, C2, MP2, S2, F1, D1, S3, D2, S4), LR)

mnist = Batcher(MNIST(), MBS)

# # ConvNet
# for x, y in mnist:
#   y = one_hot_batch(y)
#   pred = model(x)
#   loss = mse(y, pred)
#   pred_label = np.argmax(pred, axis=-1)
#   print(sum(pred_label==np.argmax(y, axis=-1))/MBS)
#   print(loss)
#   OutGrad = mse_prime(y, pred)
#   model.backward(OutGrad)
