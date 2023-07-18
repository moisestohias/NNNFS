# extra_functional.py
""" A place to store extra either not fast or slopy functional stuff """

# from scipy.signal import fftconvolve

def _fftconv2d(Z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """ Much slower than the other ones but it works"""
    ZH, ZW = Z.shape
    KH, KW = W.shape
    pad_h = KH - 1
    pad_w = KW - 1
    ZP = np.pad(Z, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    KP = np.pad(W, ((ZH-1, ZH-1), (ZW-1, ZW-1)), mode='constant')
    result = fftconvolve(ZP, KP, mode='valid')
    return result

def _einsum_conv2d(Z, W, stride=(1,1)):
    HS,WS = stride
    ZN,ZC,ZH,ZW = Z.shape
    KOC,KIC,KH,KW = W.shape
    ZOH,ZOW = (ZH-(KH-HS))//HS, (ZW-(KW-WS))//WS
    ZNs,ZCs,ZHs,ZWs = Z.strides
    str_x = as_strided(Z, (ZN, KIC, ZOH, ZOW, KH, KW), (ZNs,ZCs,ZHs*HS,ZWs*WS,ZHs,ZWs), writeable=False)
    return np.einsum('ijYXyx,kjyx -> ikYX', str_x, W).reshape(ZN, KOC, ZOH, ZOW)

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Takes in the input X, scale parameter gamma, shift parameter beta, and an epsilon value to prevent division by zero. It calculates the mean and variance along the batch and spatial dimensions, normalizes the input using these statistics, and scales and shifts the normalized data using the gamma and beta parameters. The output is the batch-normalized data.
    The mean and variance are calculated along axis 0 and 2 because in a 2D convolutional layer, the batch normalization is applied to each channel separately across all the instances in the batch.

    The input to a convolutional layer is typically a 4D tensor with the shape (batch_size, num_channels, height, width). The mean and variance are calculated along the batch and spatial dimensions (i.e., height and width), but not along the channel dimension.

    By calculating the mean and variance along axis 0 and 2, we get the mean and variance of each channel separately, which is required for batch normalization in a convolutional layer.
    """
    # Calculate mean and variance along axis 0 and 2
    mean = X.mean(axis=(0, 2), keepdims=True)
    var = X.var(axis=(0, 2), keepdims=True)

    # Normalize input data
    X_hat = (X - mean) / np.sqrt(var + eps)

    # Scale and shift normalized data
    out = gamma.reshape((1, -1, 1)) * X_hat + beta.reshape((1, -1, 1))

    return out

# ML ##################################################################
def covariance(X):
    """
    X_{MxN} -> M Sample x N features
    X.mean(axis=0) take the mean of all samples M for a each feature N
    """
    X = X - X.mean(axis=0)
    return np.divide(X.T.dot(X), len(X)-1)


def covariance_wrt(X, Y=None):
    """
    X_{MxN} -> M Sample x N features
    X.mean(axis=0) take the mean across all samples M for a each feature N
    """
    X = X - X.mean(axis=0)
    Y = Y-Y.mean(axis=0) if Y else X
    return np.divide(X.T.dot(Y), len(X)-1)

def pca(X, n_components):
    covariance_matrix = covariance(X)
     # Where (eigenvectors[:,0] corresponds to eigenvalues[0])
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Keep only n_components
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]
    return X.dot(eigenvectors) # Project the data onto principal components


