import numpy as np
from numpy.fft import fft2, ifft2
as_strided = np.lib.stride_tricks.as_strided
from time import perf_counter
from math import floor, ceil
np.set_printoptions(precision=3)

#Conv
def _pad(Z: np.ndarray, K: np.ndarray, mode: str="valid") -> np.ndarray:
    """ Check arguments and pad for conv/corr """
    if mode not in ["full", "same", "valid"]: raise ValueError("mode must be one of ['full', 'same', 'valid']")
    if Z.ndim != K.ndim: raise ValueError("Z and K must have the same number of dimensions")
    if Z.size == 0 or K.size == 0: raise ValueError(f"Zero-size arrays not supported in convolutions.")

    InputLargerThanKernel = all(s1 >= s2 for s1, s2 in zip(Z.shape[1:], K.shape[1:]))
    if not InputLargerThanKernel: raise ValueError("Input must be larger than the Kernel in every dimension, except Depth.")
    ZN,ZC,ZH,ZW = Z.shape
    OutCh,KC,KH,KW = W.shape
    if ZC!=KC: raise ValueError(f"Kernel must have the same number channels as Input, got Z.shape:{Z.shape}, W.shape {W.shape}")
    if mode == 'valid' : padding = ((0,0),(0,0), (0,0), (0,0))
    elif mode == 'same':
        # OH = ZH-KH+1 -> ZH=OH+KH-1
        PadTop, PadBottom = floor((KH-1)/2), ceil((KH-1)/2)
        PadLeft, PadRigh = floor((KW-1)/2), ceil((KW-1)/2)
        padding = ((0,0),(0,0), (PadTop, PadBottom),(PadLeft, PadRigh))
    elif mode == 'full':
        PadTop, PadBottom = KH-1, KH-1 # full-convolution aligns kernel edge with the firs pixel of input, thus K-1
        PadLeft, PadRigh = KW-1, KW-1
        padding = ((0,0),(0,0), (PadTop, PadBottom),(PadLeft, PadRigh))
    if np.array(padding).any(): Z = np.pad(Z, padding)
    return Z, K

def _corr2d(Z, W):
    """ Fastest conv in pure Numpy other implmenetation are found at the bottom
    K(ouCh, inCh, H, W)
    Z(NCHW)
    K = 10*4*3*3 -> innerDim:10*4*3*3=10,1,36 [----] # Each channel kernels represented by a single row
    Z = 10*4*8*8 -> 1*4*6*6*3*3 ->? 1         [-]  # The correpsonding small matrices & channels should be a vector
                                              [-]
                                              [-]
                                              [-]
    That's why the channels should be last after HxW
    """
    Z = Z.transpose(0,2,3,1) # NCHW -> NHWC
    W = W.transpose(2,3,1,0) # OIKK -> KKIO

    N,ZH,ZW,C_in = Z.shape
    KH,KW,_,C_out = W.shape
    Ns, ZHs, ZWs, Cs = Z.strides

    inner_dim = KH * KW * C_in # Size of kernel flattened
    A = as_strided(Z, shape = (N, ZH-KH+1, ZW-KW+1, KH, KW, C_in), strides = (Ns, ZHs, ZWs, ZHs, ZWs, Cs)).reshape(-1,inner_dim)
    out = A @ W.reshape(-1, C_out)
    return out.reshape(N,ZH-KH+1,ZW-KW+1,C_out).transpose(0,3,1,2) # NHWC -> NCHW

def conv2d(Z, W, mode:str="valid"): return _corr2d(*_pad(Z, np.flip(W), mode))
def corr2d(Z, W, mode:str="valid"): return _corr2d(*_pad(Z, W, mode))

def _fftconv2d(Z, K):
    # Pad inputs to make them the same size
    ZH, ZW = Z.shape[-2:]
    KH, KW = K.shape[-2:]
    pad_h = KH - 1
    pad_w = KW - 1
    KPTop, KPBottom, KPLeft, KPRight = (ZH-KH)//2, (ZH-KH)//2, (ZW-KW)//2, (ZW-KW)//2
    K_padded = np.pad(K, ((KPTop, KPBottom),(KPLeft, KPRight)) , mode='constant')
    print(K_padded.shape)
    raise SystemExit
    print(Z.shape, K_padded.shape)
    Z_fft = fft2(Z, axes=(0, 1))
    K_fft = fft2(K_padded, axes=(0, 1))
    Y_fft = Z_fft * K_fft # Compute element-wise multiplication in the frequency domain
    Y = np.real(ifft2(Y_fft, axes=(0, 1))) # Compute inverse FFT of result
    print(Y[KH//2:-KH//2, KW:-KW].shape)
    return Y[KH:-KH, pad_w:-pad_w] # Remove padding from output


np.random.seed(12)
Z = np.random.randn(1,1,28,28) #.astype(np.float32)
W = np.random.randn(1,1,3,3) #.astype(np.float32)
ZS, WS = Z.squeeze(),  W.squeeze()
# print(fftConvOut.shape)
# print(fftConvOut)
fftConvOut = _fftconv2d(ZS, WS)
Conv2dOut = corr2d(Z, W).squeeze()
print(fftConvOut.shape)
print(Conv2dOut.shape)
print(np.linalg.norm(Conv2dOut-fftConvOut).round(10))

# Z = np.random.randn(8,8).astype(np.float32)
# W = np.random.randn(2,2).astype(np.float32)


def _fftconv2d(Z, K):
    # Pad inputs to make them the same size
    h, w = Z.shape[0], Z.shape[1]
    kh, kw = K.shape[0], K.shape[1]
    pad_h = kh - 1
    pad_w = kw - 1
    Z_padded = np.pad(Z, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    K_padded = np.pad(K, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')

    # Compute FFTs of inputs
    Z_fft = fft2(Z_padded, axes=(0, 1))
    K_fft = fft2(K_padded, axes=(0, 1))

    # Compute element-wise multiplication in the frequency domain
    Y_fft = Z_fft * K_fft

    # Compute inverse FFT of result
    Y = np.real(ifft2(Y_fft, axes=(0, 1)))

    # Remove padding from output
    Y = Y[pad_h:-pad_h, pad_w:-pad_w, :]

    return Y

# def _fftconv2d(Z, W):
#     k_h, k_w = W.shape
#     p_h, p_w = k_h - 1, k_w - 1
#     pad_h = p_h // 2
#     pad_w = p_w // 2
#     Z_padded = np.pad(Z, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
#     out = np.zeros_like(Z_padded)
#     for x in range(Z_padded.shape[0] - k_h):
#         for y in range(Z_padded.shape[1] - k_w):
#             patch = Z_padded[x:x + k_h, y:y + k_w]
#             out[x, y] = np.sum(patch * W)
#     return out[pad_h:-pad_h, pad_w:-pad_w]

# def _fftconv2d(Z, W):
#     ZH, ZW = Z.shape
#     KH, KW = W.shape
#     pad_h = KH - 1
#     pad_w = KW - 1
#     # ZP = np.pad(Z, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
#     KP = np.pad(W, ((ZH - KH, ZH - KH), (ZW - KW, ZW - KW)), mode='constant')
#     print(Z.shape)
#     print(KP.shape)
#     raise SystemExit
#     Z_fft = np.fft.fft2(ZP)
#     W_fft = np.fft.fft2(KP)
#     ZW_fft = Z_fft * W_fft
#     ZW = np.fft.ifft2(ZW_fft)
#     ZW = ZW.real
#     ZW = ZW[pad_h:-pad_h, pad_w:-pad_w]
#     return ZW

def _fftconv2d(Z, W):
    """FFT convolution."""
    # Get input and kernel shapes
    H, W = Z.shape
    h, w, _ = W.shape

    # Calculate padding
    pad_h = (h - 1) // 2
    pad_w = (w - 1) // 2

    # Pad input and kernel
    Z_padded = np.pad(Z, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
    W_padded = np.pad(W, ((H - h, 0), (W - w, 0), (0, 0)), mode='constant')

    # Compute FFTs
    Z_fft = np.fft.fft2(Z_padded, axes=(0, 1))
    W_fft = np.fft.fft2(W_padded, axes=(0, 1))

    # Compute element-wise multiplication in Fourier domain
    Y_fft = Z_fft * W_fft

    # Compute IFFT
    Y = np.real(np.fft.ifft2(Y_fft, axes=(0, 1)))

    # Crop output to original size
    Y = Y[pad_h:pad_h + H, pad_w:pad_w + W, :]

    return Y


def _fftconv2d(Z: np.ndarray, K: np.ndarray) -> np.ndarray:
    ZH, ZW = Z.shape
    KH, KW = K.shape
    pad_h = (KH - 1) // 2
    pad_w = (KW - 1) // 2
    Z_padded = np.pad(Z, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    K_padded = np.zeros((ZH + 2*pad_h, ZW + 2*pad_w))
    K_padded[:KH, :KW] = K
    K_flip = np.flip(np.flip(K_padded, 0), 1)
    Z_col = as_strided(Z_padded, shape=(ZH, ZW, KH, KW), strides=Z_padded.strides + Z_padded.strides)
    print(K_padded)
    print(Z.shape)
    print(Z_padded.shape)
    print(K_padded.shape)
    print(Z_col.shape)
    raise SystemExit
    Z_col = np.ascontiguousarray(np.einsum('ijklm->lmijk', Z_col))
    # print(Z_col.shape)
    K_col = np.ascontiguousarray(K_flip.reshape((KH*KW*3, 1)))
    out = np.einsum('ij,ijk->ijk', K_col, Z_col)
    out = np.sum(out, axis=(1, 2))
    return out

def convolve(Z, W, stride=1):
    # Get shapes and dimensions
    ZN, ZC, ZH, ZW = Z.shape
    K, _, kH, kW = sst.shape

    # Pad input tensor
    pad = ((0, 0), (0, 0), (kH // 2, kH // 2), (kW // 2, kW // 2))
    Z = np.pad(Z, pad, mode='constant')

    # Compute output dimensions
    outH = (ZH - kH + 2 * (kH // 2)) // stride + 1
    outW = (ZW - kW + 2 * (kW // 2)) // stride + 1

    # Create view of input tensor for sliding kernel
    shp = (ZN, ZC, outH, outW, kH, kW)
    strides = (Z.strides[0], Z.strides[1], Z.strides[2]*stride, Z.strides[3]*stride, Z.strides[2], Z.strides[3])
    view = as_strided(Z, shape=shp, strides=strides)

    # Reshape input view and kernel for einsum
    X = np.ascontiguousarray(viesst.reshape(ZN*outH*outW, C*kH*kW))
    W = np.ascontiguousarray(sst.reshape(K, ZC*kH*kW))

    # Compute convolution
    output = np.einsum('ij,kj->ik', X, sst).reshape(N, K, outH, outW)
    return output

def _convolve(Z, sst, stride=1):
    # Get shapes and dimensions
    ZN, ZC, ZH, ZW = Z.shape
    K, _, KH, KW = sst.shape

    # Pad input tensor
    # pad = ((0, 0), (0, 0), (KH // 2, KH // 2), (KW // 2, KW // 2))
    # Z = np.pad(Z, pad, mode='constant')

    # Compute output dimensions
    outH = (ZH - KH + 2 * (KH // 2)) // stride + 1
    outW = (ZW - KW + 2 * (KW // 2)) // stride + 1

    # Create view of input tensor for sliding kernel
    shp = (ZN, ZC, outH, outW, KH, KW)
    strides = (Z.strides[0], Z.strides[1], Z.strides[2]*stride, Z.strides[3]*stride, Z.strides[2], Z.strides[3])
    view = as_strided(Z, shape=shp, strides=strides)

    # Reshape input view and kernel for einsum
    X = np.ascontiguousarray(view.reshape(ZN*outH*outW, ZC*KH*KW))
    W = np.ascontiguousarray(sst.reshape(K, ZC*KH*KW))

    # Compute convolution
    output = np.einsum('ij,kj->ik', X, W).reshape(ZN, K, outH, outW)

    return output
