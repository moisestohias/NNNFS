# NeeDL 🪡 (NEcessary Elements of DL)
The **tiniest & fastest** 🔥 DL framework in **pure Numpy** 💨

>⚠️ Warning: Work in progress.
## TODO
- [ ] change maxpool2d: maxpool2dP can be achieved with single loop with im2col, check ManyFacedGodMaxPooling.py
- [ ] implement optimizers class
- [ ] implement pytorch style model
- [ ] Implement LSTM-backward
- [ ] Implement RNN

+ What is this? My attempt to write a **tiny & fast** comprehensive DL framework from scratch in **pure Numpy**.
+ Why? Golden Rule: You wont understand how something work till you make it by yourself.
+ **Why this is the best**? After I started working on this project, I've found there are many other people already worked on this same idea (shit probably I need to site them all for credibility), but as I've digged into the code, to see how they are implmening things, I found that they are either:
    + Extremeley slow, due to use of loops and efficient use of memory.
    + The code base is really complicated and much long then it needs to be.
    + They deviating from the standard convention of beloved libararies (Pytorch ofcourse), this include naming convention, arguments,  way of building models...
    + Incomplete
    + ...

# Design philosophy:
+ Efficiency: **Efficiency is the star of the show**, *by my account this is the fastest 🐇 Pure Numpy DL framework*, all layers are optimized for batching.
+ Simplicity & Succinctness: The code must remain clean, simple, succinct and  easy to read (if you know what's going on) as much as possible, no need for doc 😁
+ Modularity: Everything should be modular and independent, this goes for Layers, Activations functions, Losses, Optimizers...

# Convention:
z.shape: (N,C,H,W)
N: Input Samples (aka batch)
C: Channels
H: Height
W: Width


## Naming convention:
Fuck, we need to work on this
z: input
topGrad: zGrad of next layer
zGrad: topGrad

MBS: MiniBatchSize
LR: learning_rate
inF, outF, inShape, outShape,
+ Functional: mse/mseP, cross_entropy/cross_entropyP/
    p: y_pred : predicted/probability
    y: y_truth: target 


# Layers:
`Linear(inF, outF, bias)`
outShape
layers_name

## Main Components:
<!-- NN are just a stack of diffrentiable functions,  This is not autodiff engine, this backprop framework -->
+ **Network**: A container for the layers, takes in: optimizes, loss
+ **Layers**: Similar to any other DL framework, **Layers** are the main building block, and everything is treated as, this one is no exception, everything is treated as layer, whether it's concrete NN layer (Linear/Conv2D/Pool...), utility layer (Dropout/Reshape) or Activation Function (The non-linearity part of the NN layer). 
    + NN layers: Linear, Conv1d, Conv2d, MaxPool2d, Droput, BatchNorm, RNN, LSTM ... 
    + Activation: Relu, LeakyRelu, Sigmoid, Tanh...
!Note: the reason behind formulating the activation as layers, to make the gradient calculation much simpler and cleaner.
+ **Losses**: for measuring the error.
+ **Optimzers**: Responsible for updating layers parameters.


## Layers:

+ NN concrete layers should be simple, they should only perform basic linear operations, the activation layer is the part responsible for the non-linearity during the forward pass. 
Each NN concrete layer, must define two fundamental methods:
    + forward : Take an input perform linear part returns output
    + backward: Take the Accumulated Gradient (AG) from next layer, calculate the gradient w.r.t its weight and biases and passes them to the optimizer.
    return accumulated gradient (aka error). 

## Activation function: are treated as layer. They are responsible for:
    + The non-linearity during the backward pass
    + Calculate the accumulated gradient during the backward pass.

## Weights:
Doesn't matter how they are represented, you're going to perform the transpose during the backward pass anyway. I Like Micheal's implementation, basically rows represent nodes in the current layer.

## Optimizers:
The optimization part is performed by the optimizer (updating the net parameters )



## Supported Layers:
Linear, Conv2d, (Conv1D to be added), BatchNorm, Dropout, Attention, Reshape, Flatten..
>Notes: All layers are batched


### Inspiration:
+ [ML-From-scratch](github.com/eriklindernoren/ML-From-Scratch)
+ [python-neural-networks ](github.com/OmarAflak/python-neural-networks)

### Challenges:
+ Before I dicided to follow pytorch convention, 😶 the biggest challenge was not just how to implement operations (but honnestly yes it is), but how to organize the code, such that the implementation won't come back to bite you later down the line.


## Guidlines:
+ \__init\__ methods should never raise NotImplmentedErrro 
+ I don't wanna bother too much with type hinting, only use it where it's easy. 
+ The code must remain simple, I don't want to deal with exception and error handling.., and also makes the translate of code to anothr library easy.

## TIPs:
+ Use method instead of function call whenever possible for faster computation.

## Helper function:
As it's always the case you're going to need helper functions, since there will be many utils helper function, they should be organized into data manipulation utils, and misc utils.


```python
from MoisesNNFS.layers import Linear, Reshape
from MoisesNNFS.activatios import Tanh
from MoisesNNFS.losses import MSE
from MoisesNNFS.opimizers import SGD
from MoisesNNFS import Network
from MoisesNNFS.utils import MNIST, Batcher

optim = SGD(LR=0.001, epochs=20, MBS=32)
criterian = MSE()
model = Network(optim, )
model.add(Linear(784, 100))
model.add(Tanh())
model.add(Linear(100, 10))
model.add(Sigmoid())

_mnist = MNIST(OneHot=True)
for i in range(epochs):
    mnist = Batcher(_mnist, MBS)
    for x, y in mnist:
        pred = model(x)
        ...

```


# Guidlines: 
+ __init__ methods should never raise NotImplmentedErrro 
+ I don't wanna bother too much with type hinting, only use it where it's easy. 
+ The code simple must remain simple, I don't want to deal with exception and error handling.., and also makes the translate of code to anothr library easy.

# TIPs: 
+ Use method instead of function call whenever possible for faster computation.

## Helper function: As it's always the case you're going to need helper functions, 


🏆 Layers: are the main building block of any NN framework, this one is no exception, everything is treated as layer, whether it's concrete NN layer (Dense/Conv2D/Pool...), utility layer (Dropout/Reshape) or Activation Function (The non-linearity part of the NN layer). 

+ NN concrete layers should be simple, they should only perform basic linear operations, the activation layer is the part responsible for the non-linearity during the forward pass. 
Each NN concrete layer, must define two fundamental methods:
    forward : Take an input perform linear part returns output
    backward: Take the Accumulated Gradient (AG) from next layer, calculate the gradient w.r.t its weight and biases and passes them to the optimizer.
    return accumulated gradient (aka error). 


+ Weights: Doesn't matter how they are represented, you're going to perform the transpose during the backward pass anyway. I Like Micheal's implementation, basically rows represent nodes in the current layer.

+ The optimization part is performed by the optimizer (updating the net parameters )

### Artifacts: littl notes I've left when I initiated this projects, -kind of cute- so I am leaving it to remind my self how things started and how *cute* they were.
This document, represents my thought and ideas during the process of thinking about the implementation of NN framework from scratch using only Numpy, even though I have access to other people code, who have done the same thing before, but it's still hard, I am sure part of this is because this is the first time I've decided to work on something kind bigger than what I used to, which are basically just a single file scripts. So "It will get easier". 
Another reason for why this is still hard, is because is just me overthinking it trying to account for many situations... Seriously creating a framework is hard, especially if you are a beginner.


```py
a = z@w
da/dw = z.T # wGrad = z.T.dot(topGrad)
da/dz = w.T # zGrad = np.dot(topGrad, w.T)

```