```python
class BatchNorm(Layer):
    def __init__(self, dim, num_features, eps=1e-05, momentum=0.1):
        self.layers_name = self.__class__.__name__
        self.trainable = True
        self.training = True # Training phase 
        self.momentum = momentum
        self.eps = eps
        self.gama, self.beta = np.ones(dim), np.zeros(dim) # aka BatchNorm_gain, BatchNorm_bias
        self.runMean, self.runVar = np.zeros(dim), np.ones(dim) # Both runMean/runVar are used for infrence only
        self.params = (self.gama, self.beta)

    def forward(self, input):
        if not self.training: return self.gama*(input-self.runMean)/self.runVar + self.beta
        self.input = input
        self.batchMean, self.batchVar = input.mean(0, keepdims=True), input.var(0, keepdims=True)
        self.runMean = self.runMean*(1-self.momentum) + self.batchMean*self.momentum # Only to be used during infr
        self.runVar = self.runVar*(1-self.momentum) + self.batchVar*self.momentum # Only to be used during infr
        zhat = (input-self.batchMean)/np.sqrt(self.batchVar+self.eps)
        return self.gama*zhat+self.beta

    def backward(self, TopGrad):
        ...
```
The above code is implementing the batchNorm layer used in DL, I want you to implement the backward metho that calculates the gradient, step-by-step reason through your ideas, make sure the implementation is correct, and effecient

The above code is implementing the batchNorm layer used in DL, I want you to implement the batchNormBackward function that calculates the gradient, step-by-step reason through your ideas, make sure the implementation is correct, and effecient

```py
def batchNormBackward(TopGrad, batchMean, batchVar, gama, beta):
    N = TopGrad.shape[0]
    Zhat = (TopGrad * gama)
    Zhat_batchMean = Zhat / np.sqrt(batchVar)
    Zhat_batchVar = np.sum(Zhat * (TopGrad * gama) * (-0.5) * ((batchVar + 1e-8) ** (-1.5)), axis=0)
    Zhat_batchVar_X = 2 * (TopGrad * gama - batchMean / N) / N
    Zgrad = Zhat_batchMean + Zhat_batchVar * Zhat_batchVar_X
    BNgainGrad = np.sum(TopGrad * (TopGrad - batchMean) / np.sqrt(batchVar), axis=0)
    BNbiasGrad = np.sum(TopGrad, axis=0)
    return Zgrad, BNgainGrad, BNbiasGrad
```
