# Resources.md

$$Hout = \lfloor \frac{Hin +2∗padding[0]−dilation[0]×(kernelSize[0]−1)−1}{stride[0]} +1 \rfloor$$
$$Wout = \lfloor \frac{Win +2∗padding[1]−dilation[1]×(kernelSize[1]−1)−1}{stride[1]} +1 \rfloor$$


Pytorch's max_unpool2d function is used to reverse the effects of max_pool2d. This is done by taking the maximum value from the pooling window, and copying it to the corresponding position in the original input tensor.

The equivalent implementation in pure numpy without for loops can be written using the as_strided function, which allows the creation of a view of the array with a different shape and strides.

```python
def max_unpool2d(input, indices, output_size):
    output_shape = (input.shape[0], output_size[0], output_size[1], input.shape[3])
    strides = (input.strides[0], input.strides[1], input.strides[2], input.strides[3])
    output = np.zeros(output_shape, dtype=input.dtype)
    output = np.as_strided(output, strides=strides, shape=output_shape)
    output[indices[0], indices[1], indices[2], indices[3]] = input
    return output
```
stackoverflow.com/questions/41699513/how-to-update-the-weights-of-a-deconvolutional-layer
stackoverflow.com/questions/34254679/how-can-i-implement-deconvolution-layer-for-a-cnn-in-numpy
stackoverflow.com/questions/40615034/understanding-scipy-deconvolve
docs.scipy.org/doc/scipy/reference/generated/scipy.signal.deconvolve.html
stackoverflow.com/questions/41699513/how-to-update-the-weights-of-a-deconvolutional-layer
github.com/many-facedgod/Numpy-Atrous-Transposed-CNN
towardsdatascience.com/advanced-numpy-master-stride-tricks-with-25-illustrated-exercises-923a9393ab20
remykarem.medium.com/
medium.com/analytics-vidhya/simple-cnn-using-numpy-part-iii-relu-max-pooling-softmax-c03a3377eaf2
stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
nbviewer.org/github/craffel/crucialpython/blob/master/week3/stride_tricks.ipynb
