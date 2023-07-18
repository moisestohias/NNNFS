#ManyFacedGodConv.py
import numpy as np
as_strided = np.lib.stride_tricks.as_strided


def maxpool2d(image, pool_shape, dilation=1, stride=None):
    """
    Performs the max-pooling operation on the image
    :param image: The image to be maxpooled
    :param pool_shape: The shape of the pool filter
    :param dilation: The dilation of the filter
    :param stride: The stride for the filter (defaults to the shape of the pool
    :return: The pooled image and the argmax cache used for backprop as a tuple
    """
    if stride is None: stride = pool_shape
    im_shape = image.shape
    dilated_shape = ((pool_shape[0] - 1) * dilation + 1, (pool_shape[1] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    imrow = _im_to_rows(image, (1, im_shape[1]) + pool_shape, dilation, stride, dilated_shape, res_shape)
    imrow = imrow.reshape((imrow.shape[0], imrow.shape[1], im_shape[1], -1))
    maxpooled = np.max(imrow, axis=3).transpose((0, 2, 1))
    maxpooled = maxpooled.reshape((maxpooled.shape[0], maxpooled.shape[1], res_shape[0], res_shape[1]))
    max_indices = np.argmax(imrow, axis=3)
    return maxpooled, max_indices

def _im_to_rows(x, filter_shape, dilation, stride, dilated_shape, res_shape):
    """
    Converts the 4D image to a form such that convolution can be performed via matrix multiplication
    :param x: The image of the dimensions (batch, channels, height, width)
    :param filter_shape: The shape of the filter (num_filters, depth, height, width)
    :param dilation: The dilation for the filter
    :param stride: The stride for the filter
    :param dilated_shape: The dilated shape of the filter
    :param res_shape: The shape of the expected result
    :return: The transformed image
    """
    dilated_rows, dilated_cols = dilated_shape
    num_rows, num_cols = res_shape
    res = np.zeros((x.shape[0], num_rows * num_cols, filter_shape[1], filter_shape[2], filter_shape[3]), dtype=x.dtype)
    for i in range(num_rows):
        for j in range(num_cols):
            res[:, i * num_cols + j, :, :, :] = x[:, :, i * stride[0]:i * stride[0] + dilated_rows:dilation, j * stride[1]:j * stride[1] + dilated_cols:dilation]
    return res.reshape((res.shape[0], res.shape[1], -1))

def backward_maxpool2d(top_grad, max_indices, image, pool_shape, dilation=1, stride=None):
    """
    Performs the backward pass on the max-pool operation
    :param top_grad: The grad from the next op
    :param max_indices: The cache generated in the forward pass
    :param image: The original input image to this op
    :param pool_shape: The shape of the max-pool
    :param dilation: The dilation factor
    :param stride: The stride for the pool (defaults to the shape of the pool)
    :return: The gradient wrt the input image
    """
    if stride is None:
        stride = pool_shape
    im_shape = image.shape
    dilated_shape = ((pool_shape[0] - 1) * dilation + 1, (pool_shape[1] - 1) * dilation + 1)
    res_shape = ((im_shape[2] - dilated_shape[0]) // stride[0] + 1, (im_shape[3] - dilated_shape[1]) // stride[1] + 1)
    gradrow = np.zeros((im_shape[0], res_shape[0] * res_shape[1], im_shape[1], pool_shape[0] * pool_shape[1]), dtype=top_grad.dtype)
    gradmat = top_grad.reshape((top_grad.shape[0], top_grad.shape[1], -1)).transpose((0, 2, 1))
    i1, i2, i3 = np.ogrid[:image.shape[0], :res_shape[0] * res_shape[1], :im_shape[1]]
    gradrow[i1, i2, i3, max_indices] = gradmat
    inp_grad = _backward_im_to_rows(gradrow, image.shape, (1, im_shape[1]) + pool_shape, dilation, stride, dilated_shape, res_shape)
    return inp_grad


def _backward_im_to_rows(top_grad, inp_shape, filter_shape, dilation, stride, dilated_shape, res_shape):
    """
    Gradient transformation for the im2rows operation
    :param top_grad: The grad from the next layer
    :param inp_shape: The shape of the input image
    :param filter_shape: The shape of the filter (num_filters, depth, height, width)
    :param dilation: The dilation for the filter
    :param stride: The stride for the filter
    :param dilated_shape: The dilated shape of the filter
    :param res_shape: The shape of the expected result
    :return: The reformed gradient of the shape of the image
    """
    dilated_rows, dilated_cols = dilated_shape
    num_rows, num_cols = res_shape
    res = np.zeros(inp_shape, dtype=top_grad.dtype)
    top_grad = top_grad.reshape(
        (top_grad.shape[0], top_grad.shape[1], filter_shape[1], filter_shape[2], filter_shape[3]))
    for it in range(num_rows * num_cols):
        i = it // num_rows
        j = it % num_rows
        res[:, :, i * stride[0]:i * stride[0] + dilated_rows:dilation,
            j * stride[1]:j * stride[1] + dilated_cols:dilation] += top_grad[:, it, :, :, :]
    return res



np.random.seed(12)
Z = np.random.randint(1,10, (2,2,6,6)).astype(np.float32)
ZP, Indx  = maxpool2d(Z, (2,2))
top_grad = np.random.randn(*ZP.shape)
ZUnp = backward_maxpool2d(top_grad, Indx, Z, (2,2))
print(Z.shape)
print(ZP.shape)
print(Indx.shape)
print(ZUnp.shape)
