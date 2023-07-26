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

def insert_zeros(a, row, col):
    for i in range(1, row+1): a = np.insert(a, np.arange(1, a.shape[1], i), 0, axis=0)
    for i in range(1, col+1): a = np.insert(a, np.arange(1, a.shape[1], i), 0, axis=1)
    return a

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

