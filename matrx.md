#### Matrix multiplication derivative: `a = z@w`


$$A_{m\times n} = Z_{m\times k}\times W_{k\times  n}$$


$$
\overbrace{
\begin{bmatrix}
a_{1,1} & a_{1,2} & ... & a_{1,n} \\
\vdots  & \vdots & \vdots & \vdots \\
a_{m,1} & a_{m,1} & ... & a_{m,n} \\
\end{bmatrix}
}^{A}

=

\overbrace{
\begin{bmatrix}
z_{1,1} & ... & z_{1,k} \\
\vdots  & \vdots & \vdots \\
z_{m,1} & ... & z_{m,k} \\
\end{bmatrix}
}^{Z}

\times

\overbrace{
\begin{bmatrix}
w_{1,1} & w_{1,2} & ... & w_{1,n} \\
\vdots  & \vdots & \vdots & \vdots \\
w_{k,1} & w_{k,1} & ... & w_{k,n} \\
\end{bmatrix}
}^{W}
$$

---
#### Let's assum simple example, to be able to derive by hand
Let's assum we have a matrix
$$A = \begin{bmatrix}
a_{1,1} & a_{1,2} \\
a_{2,1} & a_{2,2}
\end{bmatrix}$$
which is the output of the current layer, and we have gradient of the loss with respect of the output of that layer $\frac{\partial L}{\partial A} $.

$$
\frac{\partial L}{\partial A} =
\begin{bmatrix}
\frac{\partial L}{a_{1,1}} & \frac{\partial L}{a_{1,2}} \\ \frac{\partial L}{a_{2,1}} & \frac{\partial L}{a_{2,2}} \end{bmatrix}
$$

$$
\overbrace{
\begin{bmatrix}
a_{1,1} & a_{1,2}  \\
a_{2,1} & a_{2,2}  \\
\end{bmatrix}
}^{A}

=

\overbrace{
\begin{bmatrix}
z_{1,1} & z_{1,2}  \\
z_{2,1} & z_{2,2}  \\
\end{bmatrix}
}^{Z}

\times

\overbrace{
\begin{bmatrix}
w_{1,1} & w_{1,2}  \\
w_{2,1} & w_{2,2}  \\
\end{bmatrix}
}^{W}
$$

Now we want to compute the gradient of the loss with respect the weight matrix $W$ $\frac{\partial L}{\partial w}$, we also need to compute the gradient of the Loss with respect to the input $\frac{\partial A}{\partial z}$, in order to continue pushing the gradient backward (hence the name *Backpropagation*). We'll start with the gradient with $\frac{\partial L}{\partial Z}$

first we need to look how each element of $A$ is computed:

$$ A =
\begin{bmatrix}
a_{1,1},  a_{1,2} \\
a_{2,1} a_{2,2} \\
\end{bmatrix}
=
\begin{bmatrix}
z_{1,1} \times w_{1,1} + z_{1,2}\times w_{2,1} & z_{1,1} \times w_{1,2} + z_{1,2}\times w_{2,2} \\
z_{2,1} \times w_{1,1} + z_{2,2}\times w_{2,1}  & z_{2,1} \times w_{1,2} + z_{2,2}\times w_{2,2}
\end{bmatrix}
$$

$$ a_{1,1} = z_{1,1} \times w_{1,1} + z_{1,2}\times w_{2,1} $$
$$ a_{1,2} = z_{1,1} \times w_{1,2} + z_{1,2}\times w_{2,2} $$
$$ a_{2,1} = z_{2,1} \times w_{1,1} + z_{2,2}\times w_{2,1} $$
$$ a_{2,2} = z_{2,1} \times w_{1,2} + z_{2,2}\times w_{2,2} $$

Note how $z_{1,1}$ is used to compute $a_{1,1}$ and $a_{1,2}$ thus when calculating the gradient of the loss $L$ with respect to it, we need to add the its effect on all branches (nodes), in this situation we only have two.

$$ \frac{\partial L}{z_{1,1}} = \frac{\partial L}{a_{1,1}} w_{1,1} + \frac{\partial L}{a_{1,2}} w_{1,2} $$
$$ \frac{\partial L}{z_{1,2}} = \frac{\partial L}{a_{1,1}} w_{2,1} + \frac{\partial L}{a_{1,2}} w_{2,2} $$
$$ \frac{\partial L}{z_{2,1}} = \frac{\partial L}{a_{2,1}} w_{1,1} + \frac{\partial L}{a_{2,2}} w_{1,2} $$
$$ \frac{\partial L}{z_{2,2}} = \frac{\partial L}{a_{2,1}} w_{2,1} + \frac{\partial L}{a_{2,2}} w_{2,2} $$

$$
\frac{\partial L}{\partial Z} =
\begin{bmatrix}
\frac{\partial L}{z_{1,1}} & \frac{\partial L}{z_{1,2}} \\
\frac{\partial L}{z_{2,1}} & \frac{\partial L}{z_{2,2}}
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial L}{a_{1,1}} w_{1,1} + \frac{\partial L}{a_{1,2}} w_{1,2} &
\frac{\partial L}{a_{2,1}} w_{1,1} + \frac{\partial L}{a_{2,2}} w_{1,2} \\
\frac{\partial L}{a_{2,1}} w_{2,1} + \frac{\partial L}{a_{2,2}} w_{2,2} &
\frac{\partial L}{a_{1,1}} w_{2,1} + \frac{\partial L}{a_{1,2}} w_{2,2}
\end{bmatrix}
$$
We can refactor this matrix into a matrix multiplcation.

$$
\underbrace{
\begin{bmatrix}
\frac{\partial L}{a_{1,1}} & \frac{\partial L}{a_{1,2}} \\ \frac{\partial L}{a_{2,1}} & \frac{\partial L}{a_{2,2}}
\end{bmatrix}
}_{\frac{\partial L}{\partial A}}

\times

\underbrace{
\begin{bmatrix}
w_{1,1} & {w_{2,1}}  \\
w_{1,2} & w_{2,2}  \\
\end{bmatrix}
}_{W^T}
$$

>Note: that the second matrix is $W$ just tranposed

We can generalize this and write:
$$ \frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \times W^T $$
Which is what's known as the bottom grad, or the input gradient.

Now we will compute the gradient of the loss with respect of the weight matrix $W$ term-by-term:


$$ \frac{\partial L}{\partial w_{1,1}} = \frac{\partial L}{\partial a_{1,1}} z_{1,1} + \frac{\partial L}{\partial a_{2,1}} z_{2,1} $$

$$ \frac{\partial L}{\partial w_{1,2}} = \frac{\partial L}{\partial a_{1,1}} z_{1,2} + \frac{\partial L}{\partial a_{2,1}} z_{2,2} $$

$$ \frac{\partial L}{\partial w_{2,1}} = \frac{\partial L}{\partial a_{1,2}} z_{1,1} + \frac{\partial L}{\partial a_{2,2}} z_{2,1} $$

$$ \frac{\partial L}{\partial w_{2,2}} = \frac{\partial L}{\partial a_{1,2}} z_{1,2} + \frac{\partial L}{\partial a_{2,2}} z_{2,2} $$

We can put this in a matrix form:

$$
\begin{bmatrix}
\frac{\partial L}{\partial w_{1,1}} & \frac{\partial L}{\partial w_{1,2}} \\
\frac{\partial L}{\partial w_{2,1}} & \frac{\partial L}{\partial w_{2,2}} \\
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial L}{\partial a_{1,1}} z_{1,1} + \frac{\partial L}{\partial a_{2,1}} z_{2,1} &
\frac{\partial L}{\partial a_{1,1}} z_{1,2} + \frac{\partial L}{\partial a_{2,1}} z_{2,2} \\
\frac{\partial L}{\partial a_{1,2}} z_{1,1} + \frac{\partial L}{\partial a_{2,2}} z_{2,1} &
\frac{\partial L}{\partial a_{1,2}} z_{1,2} + \frac{\partial L}{\partial a_{2,2}} z_{2,2}
\end{bmatrix}
$$
Looking close at the elemeents of $\frac{\partial L}{\partial W}$ you will note that we factorize this into matrix multiplcation

$$
\frac{\partial L}{\partial W} =

\begin{bmatrix}
\frac{\partial L}{\partial w_{1,1}} & \frac{\partial L}{\partial w_{1,2}} \\
\frac{\partial L}{\partial w_{2,1}} & \frac{\partial L}{\partial w_{2,2}}
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
z_{1,1} & z_{2,1} \\
z_{1,2} & z_{2,2}
\end{bmatrix}}_{Z^T}

\times

\underbrace{
\begin{bmatrix}
\frac{\partial L}{\partial a_{1,1}} & \frac{\partial L}{\partial a_{1,2}}  \\
\frac{\partial L}{\partial a_{2,1}} & \frac{\partial L}{\partial a_{2,2}}
\end{bmatrix}}_{\frac{\partial L}{\partial W}}
$$

Again we can generalize this:
$$ \frac{\partial L}{\partial W} = Z^T \frac{\partial L}{\partial W} $$
And this is the gradient of the loss with respect the weight matrix $W$

## Having bias doesn't change anything
In Neural networks often we add bias to the output of the affin transfomration, so we have $A = Z\times W + B$ instead, having this extra bias doesn't change anything, since it's just addition operation,

$$
\overbrace{
\begin{bmatrix}
a_{1,1} & a_{1,2}  \\
a_{2,1} & a_{2,2}  \\
\end{bmatrix}
}^{A}

=

\overbrace{
\begin{bmatrix}
z_{1,1} & z_{1,2}  \\
z_{2,1} & z_{2,2}  \\
\end{bmatrix}
}^{Z}

\times

\overbrace{
\begin{bmatrix}
w_{1,1} & w_{1,2}  \\
w_{2,1} & w_{2,2}  \\
\end{bmatrix}
}^{W}
+
\overbrace{
\begin{bmatrix}
b_{1}  \\
b_{2}  \\
\end{bmatrix}
}^{B}
$$
>Note: The bias is replcated to match the size of the matrix
$$ a_{1,1} = z_{1,1} \times w_{1,1} + z_{1,2}\times w_{2,1} + b_{1} $$
$$ a_{1,2} = z_{1,1} \times w_{1,2} + z_{1,2}\times w_{2,2} + b_{2} $$
$$ a_{2,1} = z_{2,1} \times w_{1,1} + z_{2,2}\times w_{2,1} + b_{1} $$
$$ a_{2,2} = z_{2,1} \times w_{1,2} + z_{2,2}\times w_{2,2} + b_{2} $$

---

Going back to first general case we can write:
$$
\begin{bmatrix}
\frac{\partial L}{\partial w_{1,1}} & \frac{\partial L}{\partial w_{1,2}} & ... & \frac{\partial L}{\partial w_{1,n}}\\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial L}{\partial w_{k,1}} & \frac{\partial L}{\partial w_{k,2}} & ... & \frac{\partial L}{\partial w_{k,n}}
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
z_{1,1} & z_{2,1} & ... & z_{k,1} \\
z_{1,2} & z_{2,2} & ... & z_{k,2} \\
\vdots & \vdots & \ddots & \vdots \\
z_{1,m} & z_{2,m} & ... & z_{k,m}
\end{bmatrix}}_{Z^T}

\times

\underbrace{
    \begin{bmatrix}
\frac{\partial L}{\partial a_{1,1}} & \frac{\partial L}{\partial a_{1,2}} & ... & \frac{\partial L}{\partial a_{1,n}}  \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial L}{\partial a_{m,1}} & \frac{\partial L}{\partial a_{m,2}} & ... & \frac{\partial L}{\partial a_{m,n}}
\end{bmatrix}
}_{\frac{\partial L}{\partial W}}
$$


$$ \frac{\partial L}{\partial w_{1,1}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,1}} z_{i,1} $$
$$ \frac{\partial L}{\partial w_{1,2}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,1}} z_{i,2} $$
$$ \frac{\partial L}{\partial w_{1,n}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,1}} z_{i,n} $$

$$ \frac{\partial L}{\partial w_{2,1}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,2}} z_{i,1} $$
$$ \frac{\partial L}{\partial w_{2,2}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,2}} z_{i,2} $$


---

Matrix multiplication and its derivative can be a tricky subject, but understanding the basic principles of matrix calculus can help. Let's start from basic principles and build up.

We are given:

$Y_{a\times c} = X_{a\times b} \times W_{b\times c}$

To compute the derivative of Y with respect to W, it can be useful to initially consider the elements approach. Let's consider this multiplication in terms of its elements, i.e., the elements of Y are computed as

$y_{ij}=\sum_{k=1}^b x_{ik}w_{kj}$

where the indices i,j represent the elements of each matrix.

In order to get the derivative with respect to one of the elements of W, say $w_{kj}$, we take a partial derivative of y_{ij} with respect to $w_{kj}$:

$\frac{\partial y_{ij}}{\partial w_{kj}}=x_{ik}$ when k and j are fixed, since all other terms in the sum disappear.

This gives us that the derivative with respect to one specific element, but we want the derivative with respect to the whole W.

Recall that each element of W contributes to not just one element of Y, but to many: $w_{kj}$ contributes to all $y_{ik}$, where i vary from 1 to a. Hence, by linearity of the derivative, we have

$\frac{\partial y_{ik}}{\partial W} = x_{ik}$, for every i.

Putting i as row indices and kj as column indices, we get the matrix X.

Finally, to aligning all the dimensions correctly, we need to transpose this resulting X to get $X^T$.

As mentioned, the partial derivative of $y_{ik}$ with respect to $W$ equals $x_{ik}$. When we arrange these quantities into matrices again, we consider each element $y_{ij}$ as a separate function and thus will have a matrix (a set of functions) for its derivative. Arranging these gives us the derivative of $Y$ with respect to $W$, which is a 4th order tensor of dimensions $a \times c \times b \times c$.

However, most of the times, it is more practical to deal with matrices instead of higher order tensors. Therefore, one commonly applied strategy is to flatten these arrays by applying the trace operator, relying on the identity tr(AB) = tr(BA).

We then get the so-called Frobenius inner product (the equivalence to the dot product for vectors but for matrices):

$<A, B>_{F} = tr(A^{T}B)$.

Applying the trace operator to our derivative gives us $\frac{\partial tr(Y)}{\partial W}$, i.e., the derivative of the trace of matrix Y with respect to W.

We have:

$tr(Y) = tr(XW)$,

and using the cyclical property of trace, i.e., $tr(AB) = tr(BA)$ for any matrices A and B for which the product is defined:

$tr(Y) = tr(WX^T)$.

Finally, we use that the derivative of the matrix trace w.r.t. to a matrix equals the matrix transposed:

$\frac{\partial tr(WX^T)}{\partial W} = X^T$,

So, the derivative of the output of our matrix multiplication w.r.t the matrix $W$ is the transposed of the matrix $X$.

---

The derivative of a matrix multiplication with respect to another matrix is a bit more complicated than the derivative of a scalar function.

Assuming we have the following matrix multiplication $Y_{a\times c} = X_{a\times b} \times W_{b\times c}$, the derivative of $Y$ with respect to $W$ is given by $\frac{\partial Y}{\partial W} = X^T$¹.

To understand why this is the case, we can start by computing the derivative of a scalar function with respect to a matrix. Let $f(x) = x^T A x$, where $x$ is a column vector and $A$ is a symmetric matrix. The derivative of $f$ with respect to $x$ is given by $\frac{\partial f}{\partial x} = 2Ax$.

Now let's consider the case where we have a matrix function $F(W) = XW$, where $X$ is a constant matrix and $W$ is the variable matrix. We want to compute the derivative of $F$ with respect to $W$. To do this, we can use the chain rule:

$$\frac{\partial Y}{\partial W} = \frac{\partial Y}{\partial F} \frac{\partial F}{\partial W}$$

where $Y$ is the output of the matrix multiplication. We know that $\frac{\partial Y}{\partial F} = I$, since the derivative of a matrix with respect to itself is just the identity matrix.

To compute $\frac{\partial F}{\partial W}$, we can write out the elements of $F$:

$$F_{ij} = \sum_{k=1}^b X_{ik} W_{kj}$$

Taking the derivative of this expression with respect to $W_{mn}$ gives:

$$\frac{\partial F_{ij}}{\partial W_{mn}} = X_{im} \delta_{nj}$$

where $\delta_{nj}$ is the Kronecker delta function. We can write this in matrix form as:

$$\frac{\partial F}{\partial W} = X^T$$

Substituting this into our expression for $\frac{\partial Y}{\partial W}$ gives:

$$\frac{\partial Y}{\partial W} = I \times X^T = X^T$$


(1) What is the derivative of matrix multiplication, wrt another matrix?. https://math.stackexchange.com/questions/3538154/what-is-the-derivative-of-matrix-multiplication-wrt-another-matrix.
(2) Vector, Matrix, and Tensor Derivatives - Stanford University. http://cs231n.stanford.edu/vecDerivs.pdf.
(3) Properties of the Trace and Matrix Derivatives - Stanford University. https://web.stanford.edu/~jduchi/projects/matrix_prop.pdf.

---

We start with the matrix multiplication:

$$Y_{a\times c} = X_{a\times b} \times W_{b\times c}$$

Where Y is the output matrix, X is the input matrix and W is the weight matrix.

The elementwise multiplication is defined as:

$$y_{ij} = \sum_{k=1}^b x_{ik} \times w_{kj}$$

Where $x_{ik}$ is an element of matrix X and $w_{kj}$ is an element of matrix W.

To calculate the derivative of Y with respect to W, we need to take the partial derivative of each element $y_{ij}$ with respect to each element $w_{kj}$:

$$\frac{\partial y_{ij}}{\partial w_{kj}} = x_{ik}$$

In vector notation, the partial derivative is:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{w}_{kj}} = \mathbf{x}_k$$

Where $\mathbf{y}$ is a vector of all elements in Y, $\mathbf{w}{kj}$ is a vector with element $w{kj}$ and all other elements 0, and $\mathbf{x}_k$ is the k-th column of X.

Since we want the derivative with respect to the entire matrix W, we need to take the derivative with respect to all elements $w_{kj}$, resulting in:

$$\frac{\partial \mathbf{Y}}{\partial \mathbf{W}} = \mathbf{X}^T$$

Therefore, the derivative of matrix multiplication Y with respect to the weight matrix W is the transpose of matrix X.

---

We start with the basic matrix multiplication equation:

$Y_{a\times c} = X_{a\times b} \times W_{b\times c}$

Where X and Y are known matrices and W is the matrix we want to take the derivative with respect to.

Taking the derivative of both sides with respect to an element $w_{ij}$ of W, we get:

$\frac{\partial Y_{a\times c}}{\partial w_{ij}} = \frac{\partial}{\partial w_{ij}} (X_{a\times b} \times W_{b \times c})$

Now, according to the multiplication rule for partial derivatives, we have:

$\frac{\partial}{\partial w_{ij}} (X_{a\times b} \times W_{b \times c}) = X_{a\times b} \times \frac{\partial W_{b \times c}}{\partial w_{ij}}$

Since only one element of W changes when differentiating with respect to $w_{ij}$, the derivative of W is a matrix with all zeros except for a 1 at the $ij$th position.

Therefore:

$\frac{\partial Y_{a\times c}}{\partial w_{ij}} = X_{a\times b} \times E_{ij}$

Where $E_{ij}$ is the matrix with 1 at the $ij$th position and zeros everywhere else.

In summary, the derivative of a matrix multiplication with respect to an element of the rightmost matrix is the left matrix multiplied by a matrix with a 1 at the corresponding position and zeros elsewhere

---

Let $X$ and $W$ be two invertible matrices and $Y$ be the be two invertible matrices and $Y$ be the matrix of their product, with $Y = X \times W$.  By taking the derivative with respect to matrix matrices and $Y$ be the matrix of their product, with $Y = X \times W$.  By taking the derivative with respect to matrix matrices and $Y$ be the matrix of their product, with $Y = X \times W$.  By taking the derivative with respect to matrix $W$ we would get $\frac{\partial Y}{\partial matrix of their product, with $Y = X \times W$.  By taking the derivative with respect to matrix $W$ we would get $\frac{\partial Y}{\partial W} = X^T$. This can be seen by considering the equation of a matrix product as a and $W$ be two invertible matrices and $Y$ be the matrix of their product, with $Y = X \times W$.  By taking the derivative with respect to matrix $W$ we would get $\frac{\partial Y}{\partial W} = X^T$. This can be seen by considering the equation of a matrix product as a function:

$Y = f(X, two invertible matrices and $Y$ be the matrix of their product, with $Y = X \times W$.  By taking the derivative with respect to matrix $W$ we would get $\frac{\partial Y}{\partial W} = X^T$. This can be seen by considering the equation of a matrix product as a function:

$Y = f(X, W) = X \times W$

and and $Y$ be the matrix of their product, with $Y = X \times W$.  By taking the derivative with respect to matrix $W$ we would get $\frac{\partial Y}{\partial W} = X^T$. This can be seen by considering the equation of a matrix product as a function:

$Y = f(X, W) = X \times W$

and taking the derivative with respect to $W$:

$\frac{\partial  By taking the derivative with respect to matrix $W$ we would get $\frac{\partial Y}{\partial W} = X^T$. This can be seen by considering the equation of a matrix product as a function:

$Y = f(X, W) = X \times W$

and taking the derivative with respect to $W$:

$\frac{\partial Y}{\partial W} = \frac{\partial (X $W$ be two invertible matrices and $Y$ be the matrix of their product, with $Y = X \times W$.  By taking the derivative with respect to matrix $W$ we would get $\frac{\partial Y}{\partial W} = X^T$. This can be seen by considering the equation of a matrix product as a function:

$Y = f(X, W) = X \times W$

and taking the derivative with respect to $W$:

$\frac{\partial Y}{\partial W} = \frac{\partial (X \times W)}{\partial W} = X \times \frac
