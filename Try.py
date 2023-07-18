# from functional import  conv2d, corr2d, corr2d_backward
import numpy as np
from math import ceil, floor
as_strided = np.lib.stride_tricks.as_strided
from functional import maxpool2d, unmaxpool2d
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch.autograd import grad
# from math import ceil, floor
# as_strided = np.lib.stride_tricks.as_strided

"""
def unmaxpool(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))
    repeater = np.ones((KH, KW))
    for n in range(ZN):
        for c in range(ZC):
            Z[n, c, :, :] = np.kron(ZP[n, c, :, :], repeater)
            ind = Indx[n, c, :, :]
            Z[n, c, :, :] *= (ind == np.arange(KH*KW)[:, None, None])
    return Z

def unmaxpool2dB(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))
    # repeat the indices along the last two dimensions
    Indx = np.repeat(np.repeat(Indx, KH, axis=2), KW, axis=3)
    # flatten the arrays
    Z = Z.reshape(-1)
    Indx = Indx.reshape(-1)
    ZP = ZP.reshape(-1)
    # use numpy.put to assign the values from ZP to Z based on the indices
    np.put(Z, Indx, ZP)
    # reshape back to the original shape
    Z = Z.reshape((ZN, ZC, ZH * KH, ZW * KW))
    return Z

def _unmaxpool2dG(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))
    # repeat the indices along the last two dimensions
    Indx = np.repeat(np.repeat(Indx, KH, axis=2), KW, axis=3) # changed axis=2 to axis=3
    # flatten the arrays
    Z = Z.reshape(-1)
    Indx = Indx.reshape(-1)
    ZP = ZP.reshape(-1)
    # use numpy.put to assign the values from ZP to Z based on the indices
    np.put(Z, Indx, ZP)
    # reshape back to the original shape
    Z = Z.reshape((ZN, ZC, ZH * KH, ZW * KW))
    return Z

def unmaxpool2dG(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))
    # repeat the indices along the last two dimensions
    Indx = np.repeat(np.repeat(Indx, KH, axis=2), KW, axis=3) # changed axis=2 to axis=3
    # flatten the arrays
    Z = Z.reshape(-1)
    Indx = Indx.reshape(-1)
    ZP = ZP.reshape(-1)
    # use numpy.put to assign the values from ZP to Z based on the indices
    np.put(Z, Indx, ZP)
    # reshape back to the original shape
    Z = Z.reshape((ZN, ZC, ZH * KH, ZW * KW))
    return Z

"""

def max_pool2d_with_indices(A, kernel_size, stride, padding=0):
  # Padding
  A = np.pad(A, padding, mode='constant')

  # Window view of A
  output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                  (A.shape[1] - kernel_size) // stride + 1)
  shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
  strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
  A_w = as_strided(A, shape_w, strides_w)

  # Return the result of pooling and the indices
  return A_w.max(axis=(2, 3)), A_w.argmax(axis=(2, 3))

def max_pool2d_with_indices(A, kernel_size, stride, padding=0):
  # Padding
  A = np.pad(A, padding, mode='constant')

  # Window view of A
  output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                  (A.shape[1] - kernel_size) // stride + 1)
  shape_w = (output_shape[0], output_shape[1], kernel_size, kernel_size)
  strides_w = (stride*A.strides[0], stride*A.strides[1], A.strides[0], A.strides[1])
  A_w = as_strided(A, shape_w, strides_w)

  # Return the result of pooling and the indices
  return A_w.max(axis=(2, 3)), np.apply_over_axes(np.argmax, A_w, axes=(2, 3)).squeeze()

def Your_unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))
    # repeat the indices along the last two dimensions
    Indx = np.repeat(np.repeat(Indx, KH, axis=2), KW, axis=3) # changed axis=2 to axis=3
    # flatten the arrays
    Z = Z.reshape(-1)
    Indx = Indx.reshape(-1)
    ZP = ZP.reshape(-1)
    # use numpy.put to assign the values from ZP to Z based on the indices
    np.put(Z, Indx, ZP)
    # reshape back to the original shape
    Z = Z.reshape((ZN, ZC, ZH * KH, ZW * KW))
    return Z

Z = np.random.randint(0,10,(1,1,6,6))
ZP, Indx = maxpool2d(Z)
ZP, Indx = max_pool2d_with_indices(Z, 2, 1)
# ZPUn = My_unmaxpool2d(ZP, Indx)
# ZPUnC = unmaxpool2d(ZP, Indx)
print(Z)
print(ZP)
# print(ZPUn)
# print(ZPUnC)


# def tanh(x): return np.tanh(x) # or 2.0*(σ((2.0 * x)))-1.0
# def tanh_prime(x): return 1 - np.tanh(x) ** 2

# RNN: (T,N,L) -> (N, H) -> (N, O)
# N, T, L, H, O = 2,5,10, 5, 3
# Z = np.random.randn(T,N,L)
# Hid = np.random.randn(H)
# Whh = np.random.randn(H,H)
# Wih = np.random.randn(L,H) # (H,L) * H -> H
# Why = np.random.randn(H,O)
# Bh = np.random.randn(H)
# By = np.random.randn(O)

# trnn = nn.RNN(L, H, 1)
# inp = torch.rand(T,N,L)
# HidState = torch.rand(1,N,H)
# Tout, Thid = trnn(inp, HidState)
# print(Tout.shape, Thid.shape)


# def rnn(Z, H, W_hh, W_ih, W_hy, Bh, By, actFun=tanh):
#     zh = H.dot(W_hh) + Z.dot(W_ih) + Bh
#     ht = actFun(zh)
#     yt = ht.dot(W_hy) + By
#     out = actFun(yt)
#     return ht, yt, zh, out

# def lrnn(Z, H, W_hh, W_ih, W_hy, Bh, By, actFun=tanh):
#     zh = H.dot(W_hh) + Z.dot(W_ih) + Bh
#     ht = actFun(zh)
#     return ht, yt, zh, out

# def rnn_prime(TopGrad, Z, ht, yt, zh, W_ih, W_hh, W_hy, Bh, By, actFunPrime=tanh_prime):
#     TopGrad = TopGrad * actFunPrime(yt)
#     Z_hyGrad, W_hyGrad, B_hyGrad = backward_affine_transform(TopGrad, ht, W_hy)
#     yt_Grad = Z_hyGrad * actFunPrime(zh)
#     Z_hhgrad, W_hhGrad, B_hhGrad = backward_affine_transform(Zgrad,ht,W_hh)
#     Z_ihgrad, W_ihGrad, B_ihGrad = backward_affine_transform(Zgrad,Z,W_ih)
#     BGrad = TopGrad.sum(axis=0)
#     return ButtomGrad, Z_hhgrad, W_hhGrad, B_hhGrad, Z_ihgrad, W_ihGrad, B_ihGrad


# ht, yt, zh, out = rnn(inp.numpy(), HidState.numpy(), Whh, Wih, Why, Bh, By)
# print(ht.shape, yt.shape, zh.shape, out.shape)
# TopGrad = np.ones_like(out)
# rnn_prime(TopGrad, Z, Hid, Yt, zh, Wih, Whh, Why, Bh, By, tanh_prime)

"""
Z = torch.rand(5,5,8,8).to(torch.float64)
W = torch.rand(5,5,3,3).to(torch.float64)
Z.requires_grad_()
W.requires_grad_()

TC = F.conv2d(Z, W)
TopGrad = torch.ones_like(TC)
TWG = grad(TC, W, TopGrad, retain_graph=True)[0]
TZG = grad(TC, Z, TopGrad, retain_graph=True)[0]

MWG, MZG = corr2d_backward(Z.detach().numpy(), W.detach().numpy(), TopGrad.detach().numpy())
print(np.linalg.norm(TZG.numpy()-MZG).round(9))
print(np.linalg.norm(TWG.numpy()-MWG).round(9))
"""
