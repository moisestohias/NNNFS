---
marp : true
# theme: uncover
# class: invert
---

# <!--fit--> Matrix multiplication derivative


---

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
which is the output of the current layer.

---

Now we want to compute the gradient of the loss with respect the weight matrix $W$ $\frac{\partial L}{\partial w}$, we also need to compute the gradient of the Loss with respect to the input $\frac{\partial A}{\partial z}$, in order to continue pushing the gradient backward (hence the name *Backpropagation*).

we also have gradient of the loss with respect of the output of that layer $\frac{\partial L}{\partial A}$.

$$\frac{\partial L}{\partial A} = \begin{bmatrix} \frac{\partial L}{a_{1,1}} & \frac{\partial L}{a_{1,2}} \\ \frac{\partial L}{a_{2,1}} & \frac{\partial L}{a_{2,2}} \end{bmatrix}$$

---


$$\overbrace{ \begin{bmatrix} a_{1,1} & a_{1,2}  \\ a_{2,1} & a_{2,2}  \\ \end{bmatrix} }^{A} = \overbrace{ \begin{bmatrix} z_{1,1} & z_{1,2}  \\ z_{2,1} & z_{2,2}  \\ \end{bmatrix} }^{Z} \times \overbrace{ \begin{bmatrix} w_{1,1} & w_{1,2}  \\ w_{2,1} & w_{2,2}  \\ \end{bmatrix} }^{W}$$
$$ =
\begin{bmatrix}
z_{1,1} \times w_{1,1} + z_{1,2}\times w_{2,1} & z_{1,1} \times w_{1,2} + z_{1,2}\times w_{2,2} \\
z_{2,1} \times w_{1,1} + z_{2,2}\times w_{2,1}  & z_{2,1} \times w_{1,2} + z_{2,2}\times w_{2,2}
\end{bmatrix}
$$

---



Let's write the term by term in separate lines
$$ a_{1,1} = z_{1,1} \times w_{1,1} + z_{1,2}\times w_{2,1} $$
$$ a_{1,2} = z_{1,1} \times w_{1,2} + z_{1,2}\times w_{2,2} $$
$$ a_{2,1} = z_{2,1} \times w_{1,1} + z_{2,2}\times w_{2,1} $$
$$ a_{2,2} = z_{2,1} \times w_{1,2} + z_{2,2}\times w_{2,2} $$


>Note how $z_{1,1}$ is used to compute $a_{1,1}$ and $a_{1,2}$ thus when calculating the gradient of the loss $L$ with respect to it, we need to add the its effect on all branches (nodes), in this situation we only have two.

$$ \frac{\partial L}{z_{1,1}} = \frac{\partial L}{a_{1,1}} w_{1,1} + \frac{\partial L}{a_{1,2}} w_{1,2} $$

---


$$ \frac{\partial L}{z_{1,1}} = \frac{\partial L}{a_{1,1}} w_{1,1} + \frac{\partial L}{a_{1,2}} w_{1,2} $$
$$ \frac{\partial L}{z_{1,2}} = \frac{\partial L}{a_{1,1}} w_{2,1} + \frac{\partial L}{a_{1,2}} w_{2,2} $$
$$ \frac{\partial L}{z_{2,1}} = \frac{\partial L}{a_{2,1}} w_{1,1} + \frac{\partial L}{a_{2,2}} w_{1,2} $$
$$ \frac{\partial L}{z_{2,2}} = \frac{\partial L}{a_{2,1}} w_{2,1} + \frac{\partial L}{a_{2,2}} w_{2,2} $$

---

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

---

>Note: that the second matrix is $W$ just tranposed

We can generalize this and write:
$$ \frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \times W^T $$
Which is what's known as the bottom grad, or the input gradient.

Now we will compute the gradient of the loss with respect of the weight matrix $W$ term-by-term:

---

$$ \frac{\partial L}{\partial w_{1,1}} = \frac{\partial L}{\partial a_{1,1}} z_{1,1} + \frac{\partial L}{\partial a_{2,1}} z_{2,1} $$

$$ \frac{\partial L}{\partial w_{1,2}} = \frac{\partial L}{\partial a_{1,1}} z_{1,2} + \frac{\partial L}{\partial a_{2,1}} z_{2,2} $$

$$ \frac{\partial L}{\partial w_{2,1}} = \frac{\partial L}{\partial a_{1,2}} z_{1,1} + \frac{\partial L}{\partial a_{2,2}} z_{2,1} $$

$$ \frac{\partial L}{\partial w_{2,2}} = \frac{\partial L}{\partial a_{1,2}} z_{1,2} + \frac{\partial L}{\partial a_{2,2}} z_{2,2} $$

We can put this in a matrix form:

---

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

---

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

---


Again we can generalize this:
$$ \frac{\partial L}{\partial W} = Z^T \frac{\partial L}{\partial W} $$
And this is the gradient of the loss with respect the weight matrix $W$

---

## Having bias doesn't change anything
In Neural networks often we add bias to the output of the affin transfomration, so we have $A = Z\times W + B$ instead, having this extra bias doesn't change anything, since it's just addition operation,

---

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

---

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

---


$$ \frac{\partial L}{\partial w_{1,1}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,1}} z_{i,1} $$
$$ \frac{\partial L}{\partial w_{1,2}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,1}} z_{i,2} $$
$$ \frac{\partial L}{\partial w_{1,n}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,1}} z_{i,n} $$

$$ \frac{\partial L}{\partial w_{2,1}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,2}} z_{i,1} $$
$$ \frac{\partial L}{\partial w_{2,2}} = \sum_{i=1}^{m} \frac{\partial L}{\partial a_{i,2}} z_{i,2} $$


