We have three matrices $A, Z, W$  where:
$$A_{m\times n} = Z_{m\times k}\times W_{k\times  n}$$
We can have:
$$A_{m\times n}\times W_{k\times  n}^T = Z_{m\times k} $$
$$Z_{m\times k}^T \times A_{m\times n} =  W_{k\times  n}$$
### Proof
$$ A = Z \times W $$
We can multiply both sides by $W^T$:
$$ A \times W^T = Z \times W \times W^T $$

Since $W \times W^T$ is a square matrix of size $k \times k$, we can write:

$$ A \times W^T = Z \times I_k $$

where $I_k$ is the identity matrix of size $k \times k$. Therefore:

$$ A \times W^T = Z $$
Similarly, we can multiply both sides of the original equation by $Z^T$:

$$ Z^T \times A = Z^T \times Z \times W $$
Since $Z^T \times Z$ is a square matrix of size $k \times k$, we can write:
$$ Z^T \times A = I_k \times W $$
where $I_k$ is the identity matrix of size $k \times k$. Therefore:
$$ Z^T \times A = W $$

<!-- $\Rightarrow \frac{\partial A}{\partial Z} = \frac{\partial A}{\partial Z} $ -->

$$
\overbrace{
\begin{bmatrix}
\frac{\partial L}{a_{1,1}} & \frac{\partial L}{a_{1,2}} \\ \frac{\partial L}{a_{2,1}} & \frac{\partial L}{a_{2,2}}
\end{bmatrix}
}^{\frac{\partial L}{\partial A}}
=
\overbrace{
\begin{bmatrix}
\frac{\partial L}{z_{1,1}} & \frac{\partial L}{z_{1,2}} \\
\frac{\partial L}{z_{2,1}} & \frac{\partial L}{z_{2,2}}
\end{bmatrix}
}^{\frac{\partial L}{\partial Z}}

\times

\overbrace{
\begin{bmatrix}
w_{1,1} & w_{1,2}  \\
w_{2,1} & w_{2,2}  \\
\end{bmatrix}
}^{W}
$$
taking $W$ to lhs
$$
\underbrace{
\begin{bmatrix}
\frac{\partial L}{a_{1,1}} & \frac{\partial L}{a_{1,2}} \\ \frac{\partial L}{a_{2,1}} & \frac{\partial L}{a_{2,2}}
\end{bmatrix}
}_{\frac{\partial L}{\partial A}}

\times

\underbrace{
\begin{bmatrix}
w_{1,1} & w_{1,2}  \\
w_{2,1} & w_{2,2}  \\
\end{bmatrix}
}_{W^T}

=
\underbrace{
\begin{bmatrix}
\frac{\partial L}{z_{1,1}} & \frac{\partial L}{z_{1,2}} \\
\frac{\partial L}{z_{2,1}} & \frac{\partial L}{z_{2,2}}
\end{bmatrix}
}_{\frac{\partial L}{\partial Z}}
$$

