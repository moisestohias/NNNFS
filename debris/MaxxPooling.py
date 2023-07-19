import numpy as np
as_strided = np.lib.stride_tricks.as_strided

def pool2d(Z, K:tuple=(2,2)):
    """ performs the windowing, and padding if needed"""
    KH, KW = K  # Kernel Height & Width
    N, C, ZH, ZW = Z.shape # Input: NCHW Batch, Channels, Height, Width
    Ns, Cs, Hs, Ws = Z.strides
    EdgeH, EdgeW = ZH%KH, ZW%KW # How many pixels left on the edge
    if (EdgeH!=0 or EdgeW!=0): # If there are pixels left we need to pad
        PadH, PadW = KH-EdgeH, KW-EdgeW
        PadTop, PadBottom = ceil(PadH/2), floor(PadH/2)
        PadLeft, PadRight = ceil(PadW/2), floor(PadW/2)
        Z = np.pad(Z, ((0,0),(0,0), (PadTop, PadBottom), (PadLeft, PadRight)))
        N, C, ZH, ZW = Z.shape #NCHW
        Ns, Cs, Hs, Ws = Z.strides
    Zstrided = as_strided(Z, shape=(N,C,ZH//KH, ZW//KW, KH, KW), strides=(Ns, Cs, Hs*KH, Ws*KW,Hs, Ws))
    return Zstrided.reshape(N,C,ZH//KH, ZW//KW, KH*KW) # reshape to flatten pooling windows to be 1D-vector

def maxpool2d(Z, K:tuple=(2,2)):
    ZP = pool2d(Z, K)
    MxP = np.max(ZP, axis=(-1))
    Inx = np.argmax(ZP, axis=-1)
    return MxP, Inx



def unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))

    # create a mask with ones at the positions of the max values
    mask = np.zeros_like(ZP)
    mask[np.arange(ZN)[:, None, None, None], np.arange(ZC)[None, :, None, None], np.arange(ZH)[:, None, None, None], np.arange(ZW)[:, None, None], Indx] = 1

    # repeat the mask KH and KW times along the last two axes
    mask = np.repeat(np.repeat(mask, KH, axis=-2), KW, axis=-1)

    # tile the pooled values KH and KW times along the last two axes
    ZP_tiled = np.tile(np.expand_dims(np.expand_dims(ZP, axis=-1), axis=-1), (1, 1, 1, 1, KH, KW)).reshape(ZN, ZC, ZH * KH, ZW * KW)

    # scatter the pooled values back into the unpooled array using the mask
    np.putmask(Z, mask.astype(bool), ZP_tiled)

    return Z

def unmaxpool2d(ZP, Indx, K:tuple=(2,2)):
    ZN,ZC,ZH,ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN,ZC,ZH*KH,ZW*KW))
    mask = np.zeros_like(Z)
    mask[np.arange(ZN)[:, None, None], np.arange(ZC)[None, :, None], ZH_idx, ZW_idx] = 1
    # mask[np.arange(ZN)[:, None, None, None], np.arange(ZC)[None, :, None, None], np.arange(ZH)[:, None, None, None], np.arange(ZW)[:, None, None], Indx] = 1
    mask = mask.reshape(ZN, ZC, ZH, ZW, KH, KW)
    ZP = ZP[..., None, None]
    ZP = np.tile(ZP, (1, 1, 1, 1, KH, KW))
    ZP = ZP * mask
    Z = np.zeros((ZN, ZC, ZH*KH, ZW*KW))
    for h in range(ZH):
        for w in range(ZW):
            Z[:, :, h*KH:(h+1)*KH, w*KW:(w+1)*KW] += ZP[:, :, h, w]
    return Z

def unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))
    mask = np.zeros_like(Z)
    mask[np.arange(ZN)[:, None, None, None], np.arange(ZC)[None, :, None, None], np.unravel_index(Indx, (KH*KW,))[None, None, None, :], ZH_idx[:, None, None, :], ZW_idx[:, None, None, :]] = 1
    Z[np.arange(ZN)[:, None, None, None], np.arange(ZC)[None, :, None, None], ZH_idx[:, None, None, :], ZW_idx[:, None, None, :]] = ZP * mask
    return Z

def unmaxpool2d(ZP, Indx, K:tuple=(2,2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH*KH, ZW*KW))
    mask = np.zeros_like(Z)

    ZH_idx, ZW_idx = np.indices((ZH, ZW))

    mask[np.arange(ZN)[:, None, None, None], np.arange(ZC)[None, :, None, None], np.unravel_index(Indx, (KH*KW,))[None, None, None, :], ZH_idx[:, None, None, :], ZW_idx[:, None, None, :]] = 1

    Z = ZP.repeat(KH, axis=2).repeat(KW, axis=3) * mask
    return Z

def unmaxpool2d(ZP, Indx, K:tuple=(2,2)):
    # print("hello")
    ZN,ZC,ZH,ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN,ZC,ZH*KH,ZW*KW))
    ZH_idx, ZW_idx = np.indices((KH, KW))
    mask = np.zeros((ZN,ZC,ZH,KW))
    mask[np.arange(ZN)[:, None, None, None], np.arange(ZC)[None, :, None, None], np.unravel_index(Indx, (KH*KW,))[0][None, None, None, :], ZH_idx[:, None, None, :], ZW_idx[:, None, None, :]] = 1
    Zp_masked = ZP*mask
    Zp_masked_reshape = Zp_masked.reshape(ZN, ZC, ZH, KW, KH).transpose(0, 1, 2, 4, 3)
    Z_unpooled = as_strided(Z, shape=Zp_masked_reshape.shape, strides=(Zp_masked_reshape.strides[0],Zp_masked_reshape.strides[1],KH*Z.strides[2],KW*Z.strides[3],Z.strides[2],Z.strides[3]))
    np.maximum(Z_unpooled, Zp_masked_reshape, out=Z_unpooled)
    return Z

# ===========================
# def maxpool2d(Z, K: tuple = (2, 2)):
#     ZP = pool2d(Z, K)
#     # MxP = np.max(ZP, axis=(-1, -2)) # take the maximum along the last 2 dimensions
#     Inx = np.argmax(ZP, axis=-1) # get the indices of the maximum
#     return Inx

def unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K

    # create a mask of zeros with the same shape as the original input Z
    mask = np.zeros((ZN, ZC, ZH * KH, ZW * KW))

    # unravel the indices of the maximum values
    Indx_ravel = np.ravel_multi_index(Indx, (KH * KW,))

    # get the indices for the corresponding maximum values
    ZH_idx, ZW_idx = np.divmod(Indx_ravel, KW)

    # set the mask values corresponding to the maximum values to 1
    mask[np.arange(ZN)[:, None, None, None], np.arange(ZC)[None, :, None, None], ZH_idx[:, None, None, :], ZW_idx[:, None, None, :]] = 1

    # reshape the pooled array to have the same shape as the mask
    ZP_reshaped = ZP.reshape(ZN, ZC, ZH, KW, ZW, KH)

    # use np.einsum to perform the unpooling operation
    Z = np.einsum('ijklmn, io, jo -> iklo', ZP_reshaped, mask, np.ones((KH, KW)))

    return Z

def unmaxpool2d(ZP, Indx, K:tuple=(2,2)):
    ZN,ZC,ZH,ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN,ZC,ZH*KH,ZW*KW))
    for n in range(ZN):
        for c in range(ZC):
            for h in range(ZH):
                for w in range(ZW):
                    kh, kw = h % KH, w % KW
                    h_out, w_out = h // KH, w // KW
                    ind = Indx[n,c,h,w]
                    Z[n,c,h_out*KH+kh,w_out*KW+kw] = ZP[n,c,h,w,ind]
    return Z

def unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))

    for n in range(ZN):
        for c in range(ZC):
            for h in range(ZH):
                for w in range(ZW):
                    ind = Indx[n, c, h, w]  # index of max value in pooling window
                    row, col = np.unravel_index(ind, (KH, KW))
                    Z[n, c, h*KH+row, w*KW+col] = ZP[n, c, h, w]

    return Z

"""

There are two errors in the following function fix theme:
1. mask is 4 dimenstional array, and you trying to index with 5 arrays
2. you are multiplying ZP and mask, which are not the same szie, ZP is the pooled and the mask is the same size of the unpooled version.

```python
def unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    ZH_idx, ZW_idx = np.indices((KH, KW))
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))
    mask = np.zeros_like(Z)
    mask[np.arange(ZN)[:, None, None, None],
    np.arange(ZC)[None, :, None, None],
    np.unravel_index(Indx, (KH*KW,))[0][None, None, None, :].astype(np.int32),
    ZH_idx[:, None, None, :],
    ZW_idx[:, None, None, :]] = 1
    Z[np.arange(ZN)[:, None, None, None], np.arange(ZC)[None, :, None, None], ZH_idx[:, None, None, :], ZW_idx[:, None, None, :]] = ZP * mask # ZP and mask are not the same szie, ZP is the pooled
    return Z
```

"""

def unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH_pooled, ZW_pooled = ZP.shape
    KH, KW = K
    ZH_unpooled, ZW_unpooled = ZH_pooled * KH, ZW_pooled * KW
    Z = np.zeros((ZN, ZC, ZH_unpooled, ZW_unpooled))
    mask = np.zeros_like(Z)
    mask[np.arange(ZN)[:, None, None, None],
         np.arange(ZC)[None, :, None, None],
         np.unravel_index(Indx, (KH*KW,))[0][None, None, :],
         np.repeat(np.arange(ZH_pooled)[:, None], KW, axis=1)[:, :, None],
         np.tile(np.arange(ZW_pooled)[None, :], (KH, 1))[:, :, None]
         ] = 1
    ZP.repeat(KH, axis=2).repeat(KW, axis=3)
    Z = ZP.repeat(KH, axis=2).repeat(KW, axis=3) * mask
    return Z

def unmaxpool2d(ZP, Indx, K: tuple = (2, 2)):
    ZN, ZC, ZH, ZW = ZP.shape
    KH, KW = K
    ZH_idx, ZW_idx = np.indices((KH, KW))
    Z = np.zeros((ZN, ZC, ZH * KH, ZW * KW))
    mask = np.zeros((ZN, ZC, ZH, ZW))
    mask[np.arange(ZN)[:, None, None, None],
         np.arange(ZC)[None, :, None, None],
         ZH_idx[:, None, None, :],
         ZW_idx[:, None, None, :]] = (Indx == (ZH_idx * KW + ZW_idx))
    Z[np.arange(ZN)[:, None, None, None],
      np.arange(ZC)[None, :, None, None],
      ZH_idx[:, None, None, :],
      ZW_idx[:, None, None, :]] = ZP.repeat(KH, axis=2).repeat(KW, axis=3) * mask.repeat(KH, axis=2).repeat(KW, axis=3)
    # print(mask.repeat(KH, axis=2).repeat(KW, axis=3).shape)
    # print(ZP.repeat(KH, axis=2).repeat(KW, axis=3).shape)
    # raise SystemExit
    return Z


np.random.seed(12)
Z = np.random.randint(1,10, (1,1,6,6)).astype(np.float32)
ZP, Indx  = maxpool2d(Z, (2,2))
# print(Z)
# print(ZP)
# print(Indx)
Zunp = unmaxpool2d(ZP, Indx, (2,2))

# print(Z.shape)
# print(ZP.shape)
# print(Zunp.shape)
# print(Z)
# print(ZP)
# print(Zunp)

# This line "mask[np.arange(ZN)[:, None, None, None], np.arange(ZC)[None, :, None, None], np.arange(ZH)[:, None, None, None], np.arange(ZW)[:, None, None], Indx] = 1" raised the following error "IndexError: too many indices for array: array is 4-dimensional, but 5 were indexed"
