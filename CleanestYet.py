def affTrans(Z, W, B=0): return Z.dot(W.T) + B # W: (outF,inF)
def affTransP(TopGrad, Z, W):
    BGrad = TopGrad.sum(axis=0)
    WGrad = TopGrad.T.dot(Z)
    Zgrad = TopGrad.dot(W)
    return Zgrad, WGrad, BGrad

class Layer:
    """ All layers Only acccept batched input: NCHW"""
    def __call__(self, x): return self.forward(x)
    def __repr__(self): return f"{self.layers_name}(Z)"
    def forward(self, input): raise NotImplementedError
    def backward(self, TopGrad): raise NotImplementedError

class Linear(Layer):
    def __init__(self, inF, outF, bias=True):
        self.layers_name = self.__class__.__name__
        self.trainable = True
        lim = 1 / np.sqrt(inF) # Only inF used to calculate the limit, avoid saturation..
        self.w  = np.random.uniform(-lim, lim, (outF, inF)) # torch style (outF, inF)
        self.b = np.random.randn(outF) * 0.1 if bias else None
        self.params = (self.w, self.b)
        self.inShape, self.outShape = (inF,), (outF,)

    def forward(self, z):
        self.z = z
        return affTrans(self.z, self.w, self.b) # [MBS,inF][outF,inF].T -> [MBS,outF]

    def backward(self, TopGrad):
        self.zGrad, self.wGrad, self.bGrad = affTransP(TopGrad, self.z, self.w)
        return self.zGrad

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """ Forward pass for batch normalization. """
    # Compute mean and variance for the batch
    batch_mean = x.mean(axis=0)
    batch_var = x.var(axis=0)
    
    # Normalize the batch
    x_normalized = (x - batch_mean) / np.sqrt(batch_var + eps)
    
    # Scale and shift
    out = gamma * x_normalized + beta
    
    # Store for backward pass
    cache = (x, x_normalized, batch_mean, batch_var, gamma, beta, eps)
    return out, cache

def batch_norm_backward(d_out, cache):
    """ Backward pass for batch normalization. """
    x, x_normalized, batch_mean, batch_var, gamma, beta, eps = cache
    N, D = x.shape
    
    # Gradient of beta and gamma
    dbeta = d_out.sum(axis=0)
    dgamma = (d_out * x_normalized).sum(axis=0)
    
    # Gradient of normalized input
    d_x_normalized = d_out * gamma
    
    # Gradient of variance and mean
    d_var = (d_x_normalized * (x - batch_mean) * -0.5 * (batch_var + eps)**(-1.5)).sum(axis=0)
    d_mean = d_x_normalized.sum(axis=0) * -1 / np.sqrt(batch_var + eps) + d_var * -2 * (x - batch_mean).mean(axis=0)
    
    # Gradient of input
    dx = d_x_normalized / np.sqrt(batch_var + eps) + d_var * 2 * (x - batch_mean) / N + d_mean / N
    
    return dx, dgamma, dbeta

class BatchNorm(Layer):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.layers_name = self.__class__.__name__
        self.trainable = True
        
        # Initialize parameters
        self.gamma = np.ones(num_features)  # Scale
        self.beta = np.zeros(num_features)   # Shift
        
        # Running averages for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.momentum = momentum
        self.eps = eps
        self.training = True  # Flag to switch between training and inference mode

    def forward(self, x):
        if self.training:
            out, self.cache = batch_norm_forward(x, self.gamma, self.beta, self.eps)
            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * x.mean(axis=0)
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * x.var(axis=0)
        else:
            # During inference, use running mean and variance
            x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_normalized + self.beta
        return out

    def backward(self, TopGrad):
        return batch_norm_backward(TopGrad, self.cache)