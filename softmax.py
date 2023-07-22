import numpy as np
import torch
import torch.nn.functional as F


def softmax(x):
  s = np.exp(x - np.max(x))
  row_sum = np.sum()
  return np.array([np.exp(x_i) / row_sum for x_i in x])

def softmax(x): s = np.exp(x-x.max()); return s / s.sum(axis=-1, keepdims=True)
def softmaxP(s): s = s.reshape(-1, 1); return np.diagflat(s) - np.dot(s, s.T)

x = np.random.randint(1,10, (2,4)).astype(np.float32)
s = softmax(x)
ds = softmax_derivative(s)

tx = torch.as_tensor(x)
tx.requires_grad_()
with torch.autograd.set_grad_enabled(True):
  ts =  F.softmax(tx)
  TopGrad = torch.ones_like(ts)
  ds = torch.autograd.grad(ts, tx, TopGrad,  retain_graph=True)[0]

one_hot([2,3], 4)


############################################################
TWG = grad(TC, W, TopGrad, retain_graph=True)[0]
TZG = grad(TC, Z, TopGrad, retain_graph=True)[0]

# Compute the derivative of the softmax function using autograd.
with torch.autograd.set_grad_enabled(True):
  ts = F.softmax(x)
  ds = torch.autograd.grad(ts, tx)[0]

# Compute the derivative of the softmax function manually.
softmax_derivative = lambda s: s - s.pow(2).sum(-1, keepdim=True)
ds_manual = softmax_derivative(expected)

# Check that the derivative is correct.
assert torch.allclose(ds, ds_manual)


# -----

# Example of y with class indices
x = torch.randn(3, 5, requires_grad=True)
y = torch.randint(5, (3,), dtype=torch.int64)
loss = F.cross_entropy(x, y)
loss.backward()
# Example of y with class probabilities

# ================
torch.manual_seed(23)
tx = torch.randn(3, 5, requires_grad=True)
ty = torch.randn(3, 5).softmax(dim=1)
loss = F.cross_entropy(tx, ty)
loss.backward()


torch.manual_seed(23)
tx = torch.randn(3, 5, requires_grad=True)
ts = F.softmax(tx, dim=1)
TopGrad = torch.ones_like(ts)
grad(ts, tx, TopGrad, retain_graph=True)[0]


x = tx.detach().numpy()


ts = F.softmax(tx, dim=1).mean()
grad(ts, tx, retain_graph=True)[0]


def backward(self, x, probs, bp_err):
    output = np.empty(x.shape)
    for j in range(x.shape[1]):
        d_prob_over_xj = - (probs * probs[:,[j]])  # i.e. prob_k * prob_j, no matter k==j or not
        d_prob_over_xj[:,j] += probs[:,j]   # i.e. when k==j, +prob_j
        output[:,j] = np.sum(bp_err * d_prob_over_xj, axis=1)
    return output