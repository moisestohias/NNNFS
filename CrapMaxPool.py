import numpy as np
import torch.nn.functional as F
import torch
# as_strided = np.lib.stride_tricks.as_strided


def maxpool2d(x):
  my, mx = (x.shape[2]//2)*2, (x.shape[3]//2)*2
  stack = []
  xup = x[:, :, :my, :mx]
  for Y in range(2):
    for X in range(2):
      stack.append(xup[:, :, Y::2, X::2][None])
  stack = np.concatenate(stack, axis=0)
  indx = np.argmax(stack, axis=0)
  return np.max(stack, axis=0), indx, x.shape

def unmaxpool2d(ZP, indx, OrigShape, grad_output=None):
  my, mx = (OrigShape[2]//2)*2, (OrigShape[3]//2)*2
  ret = np.zeros(OrigShape, dtype=np.int16) # you must specify the dtype
  for Y in range(2):
    for X in range(2): 
      ret[:, :, Y:my:2, X:mx:2] = indx == (Y*2+X)
  return ret

np.random.seed(12)
Z = np.random.randint(1,10, (1,1,6,6)).astype(np.float32)
ZP, Indx, OutShape  = maxpool2d(Z)
# print(Z)
print(ZP.shape)
# TZP = F.max_pool2d(torch.as_tensor(Z), (2,2)).numpy()
mask = unmaxpool2d(ZP, Indx, OutShape)
print(mask.shape)
print(ZP)
print(Z)
print(mask)

Z[mask]
# top_grad = np.random.randn(*ZP.shape)
# ZUnp = backward_maxpool2d(top_grad, Indx, ZP, (2,2))

# print(ZP.shape)
# print(Indx.shape)
# print(ZUnp)
