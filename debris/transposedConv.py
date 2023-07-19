def _pad(Z: np.ndarray, K: np.ndarray, mode: str="valid") -> np.ndarray:
    """ Check arguments and pad for conv/corr """
    if mode not in ["full", "same", "valid"]: raise ValueError("mode must be one of ['full', 'same', 'valid']")
    if Z.ndim != K.ndim: raise ValueError("Z and K must have the same number of dimensions")
    if Z.size == 0 or K.size == 0: raise ValueError(f"Zero-size arrays not supported in convolutions.")
    ZN,ZC,ZH,ZW = Z.shape
    OutCh,KC,KH,KW = K.shape
    if ZC!=KC: raise ValueError(f"Kernel must have the same number channels as Input, got Z.shape:{Z.shape}, W.shape {K.shape}")
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

def calculateConvOutShape(inShape, KS, S=(1,1), P=(0,0), D=(1,1)):
    """
    KS: (KH,KW) kernel_size
    S: (SH,SW) stride
    P: (PH,PW) padding
    D: (DH,DW) dilation
    """
    Hin, Win = inShape
    Hout = floor(((Hin+2*P[0]-D[0]*(KS[0]-1)-1)/S[0]) +1)
    Wout = floor(((Win+2*P[1]-D[1]*(KS[1]-1)-1)/S[1]) +1)
    return Hout, Wout # outShape

def calculateTranspConvOutShape(inShape, KS, S=(1,1), P=(0,0), OP=(0,0), D=(1,1)):
    """
    KS: (KH,KW) kernel_size
    S: (SH,SW) stride
    P: (PH,PW) padding
    OP: (PH,PW) output_padding
    D: (DH,DW) dilation
    """
    Hin, Win = inShape
    Hout =(Hin -1)*S[0]-2*P[0]+D[0]*(KS[0]-1)+OP[0]+1
    Wout =(Win -1)*S[1]-2*P[1]+D[1]*(KS[1]-1)+OP[1]+1
    return Hout, Wout # outShape

def _conv2d(Z, W, S=(1,1), P=(0,0), D=(1,1)):
    """ Convolution with stride and padding support
    Z: (N,C_in,H,W)
    W: (C_out,C_in,KH,KW)
    S: (SH,SW) stride
    P: (PH,PW) padding
    D: (DH,DW) dilation
    The reason we permute the dimensions is to make HWC of Z align with KKI of Was the last dim.
    """
    Z = Z.transpose(0,2,3,1) # NCHW -> NHWC
    W = W.transpose(2,3,1,0) # OIKK -> KKIO
    N,ZH,ZW,C_in = Z.shape
    KH,KW,_,C_out = W.shape
    Ns, ZHs, ZWs, Cs = Z.strides
    outH, outW = calculateConvOutShape((ZH,ZW), (KH,KW), S, P, D)
    outShape = (N, outH, outW, KH, KW, C_in)
    inner_dim = KH * KW * C_in # Size of kernel flattened
    A = as_strided(Z, shape = outShape, strides = (Ns, ZHs*S[0], ZWs*S[0], ZHs*D[0], ZWs*D[1], Cs)).reshape(-1,inner_dim)
    out = A @ W.reshape(-1, C_out)
    return out.reshape(N,outH,outW,C_out).transpose(0,3,1,2) # NHWC -> NCHW
def conv2d(Z, W, mode:str="valid"): return _conv2d(*_pad(Z, W, mode))

def _transp_conv2d(Z, W, S=(1,1), P=(0,0), OP=(0,0), D=(1,1)):
    """ Transposed Convolution with stride and padding support
    Z: (N,C_in,H,W)
    W: (C_out,C_in,KH,KW)
    S: (SH,SW) stride
    P: (PH,PW) padding
    OP: (PH,PW) output_padding
    D: (DH,DW) dilation
    The reason we permute the dimensions is to make HWC of Z align with KKI of Was the last dim.
    """
    Z = Z.transpose(0,2,3,1) # NCHW -> NHWC
    W = W.transpose(2,3,1,0) # OIKK -> KKIO
    N,ZH,ZW,C_in = Z.shape
    KH,KW,_,C_out = W.shape
    Ns, ZHs, ZWs, Cs = Z.strides
    outH, outW = calculateTranspConvOutShape((ZH,ZW), (KH,KW), S, P, OP, D)
    outShape = (N, outH-KH+1+2*P[0], outW-KW+1+2*P[1], KH, KW, C_in)
    inner_dim = KH * KW * C_in # Size of kernel flattened
    A = np.zeros(outShape)
    A = as_strided(A, shape=outShape+(C_out,), strides=A.strides+(0,))
    A += Z.reshape(N,ZH,ZW,-1).transpose(0,1,2,4,3) @ W.reshape(-1,C_out)
    A = A.transpose(0,3,4,5,1,2).reshape(N,-1,C_out)
    idx = np.ravel_multi_index(np.indices((KH,KW)),(KH,KW))
    A = np.add.reduceat(A,idx,axis=1)
    return A.reshape(N,outH,outW,C_out).transpose(0,3,1,2) # NHWC -> NCHW

def transp_conv2d(Z,W,S=(1,1),P=(0,0),OP=(0,0),D=(1,1)):
  return _transp_conv2d(Z,W,S,P,D)

def transp_conv2d(Z, W, S=(1,1), P=(0,0), OP=(0,0), D=(1,1)):
    """ Transposed Convolution with stride and padding support
    Z: (N,C_in,H,W)
    W: (C_out,C_in,KH,KW)
    S: (SH,SW) stride
    P: (PH,PW) padding
    OP: (PH,PW) output_padding
    D: (DH,DW) dilation
    The reason we permute the dimensions is to make HWC of Z align with KKI of Was the last dim.
    """
    Z = Z.transpose(0,2,3,1) # NCHW -> NHWC
    W = W.transpose(2,3,1,0) # OIKK -> KKIO
    N,ZH,ZW,C_in = Z.shape
    KH,KW,_,C_out = W.shape
    Ns, ZHs, ZWs, Cs = Z.strides
    outH, outW = calculateTranspConvOutShape((ZH,ZW), (KH,KW), S, P, OP, D)
    outShape = (N, outH-KH+1+2*P[0], outW-KW+1+2*P[1], KH, KW, C_in)
    inner_dim = KH * KW * C_in # Size of kernel flattened
    A = np.zeros(outShape)
    A = as_strided(A, shape=outShape+(C_out,), strides=A.strides+(0,))
    A += Z.reshape(N,ZH,ZW,-1).transpose(0,1,2,3) @ W.reshape(-1,C_out)
    A = A.transpose(0,3,4,5,1,2).reshape(N,-1,C_out)
    idx = np.ravel_multi_index(np.indices((KH,KW)),(KH,KW))
    A = np.add.reduceat(A,idx,axis=1)
    return A.reshape(N,outH,outW,C_out).transpose(0,3,1,2) # NHWC -> NCHW

def _transp_conv2d(Z, W, S=(1,1), P=(0,0), OP=(0,0), D=(1,1)):
    """ Transposed Convolution with stride and padding support
    Z: (N,C_in,H,W)
    W: (C_out,C_in,KH,KW)
    S: (SH,SW) stride
    P: (PH,PW) padding
    OP: (PH,PW) output_padding
    D: (DH,DW) dilation
    The reason we permute the dimensions is to make HWC of Z align with KKI of Was the last dim.
    """
    Z = Z.transpose(0,2,3,1) # NCHW -> NHWC
    W = W.transpose(2,3,1,0) # OIKK -> KKIO
    N,ZH,ZW,C_in = Z.shape
    KH,KW,_,C_out = W.shape
    Ns, ZHs, ZWs, Cs = Z.strides
    outH, outW = calculateTranspConvOutShape((ZH,ZW), (KH,KW), S, P, OP, D)
    outShape = (N, outH-KH+1+2*P[0], outW-KW+1+2*P[1], KH, KW, C_in)
    inner_dim = KH * KW * C_in # Size of kernel flattened
    A = np.zeros(outShape)
    A = as_strided(A, shape=outShape+(C_out,), strides=A.strides+(0,))
    A += Z.reshape(N,ZH,ZW,-1).transpose(0,1,2,3) @ W.transpose(3,2,0,1).reshape(C_out,-1).T
    A = A.transpose(0,3,4,5,1,2).reshape(N,-1,C_out)
    idx = np.ravel_multi_index(np.indices((KH,KW)),(KH,KW))
    A = np.add.reduceat(A,idx,axis=1)
    return A.reshape(N,outH,outW,C_out).transpose(0,3,1,2) # NHWC -> NCHW

def transp_conv2d(Z,W,S=(1,1),P=(0,0),OP=(0,0),D=(1,1)):
  return _transp_conv2d(Z,W,S,P,D)

