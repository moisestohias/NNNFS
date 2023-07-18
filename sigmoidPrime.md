sigmoidPrime.md

Assuming we have the following matrix multiplication $Y_{a\times c} = X_{a\times b} \times W_{b\times c}$, why the  is the derivative of matrix multiplications with respect to the matrix $w$. Explain in details please, starting from basic equations and building up to satisfying answer


+ P: padding
+ D: Dilation
+ S: stride

$$ Hout = \lfloor \frac{ Hin+2  P[0]D[0] \times (KS[0]−1)−1+1} {S[0]} \rfloor $$
$$ Wout = \lfloor \frac{ Win+2  P[1]D[1] \times (KS[1]−1)−1+1} {S[1]} \rfloor $$

```py
Hout = floor((Hin+2*P[0]*D[0]*(KS[0]−1)−1+1)/S[0])
Wout = floor((Win+2*P[0]*D[1]*(KS[1]−1)−1+1)/S[1])
```



1. The sigmoid function is defined as $\sigma(x) = \frac{1}{1 + e^{-x}}$.
2. The derivative of the sigmoid function is $\frac{d}{dx}\sigma(x) = \sigma(x)(1 − \sigma(x))$.
3. To find the derivative of the sigmoid function, we first need to find the derivative of the denominator $(1 + e^{-x})$. Using the chain rule, we get $\frac{d}{dx}(1 + e^{-x}) = -e^{-x}$.
4. Next, we need to find the derivative of the numerator $(1)$. Using the power rule, we get $\frac{d}{dx}(1) = 0$.
5. Now, we can use the quotient rule to find the derivative of $\sigma(x)$. The quotient rule states that if $f(x) = \frac{g(x)}{h(x)}$, then $f'(x) = \frac{g'(x)h(x) - h'(x)g(x)}{[h(x)]^2}$.
6. Applying the quotient rule to $\sigma(x)$, we get $\frac{d}{dx}\sigma(x) = \frac{(0)(1 + e^{-x}) - (-e^{-x})(1)}{(1 + e^{-x})^2}$.
7. Simplifying this expression gives us $\frac{d}{dx}\sigma(x) = \frac{e^{-x}}{(1 + e^{-x})^2}$.

8. To simplify this expression, we can first multiply the numerator and denominator by $(1 + e^{-x})$ to get $\frac{d}{dx}\sigma(x) = \frac{e^{-x}(1 + e^{-x})}{(1 + e^{-x})(1 + e^{-x})}$.
9. Next, we can simplify the numerator by distributing $e^{-x}$ to get $\frac{d}{dx}\sigma(x) = \frac{e^{-x} + e^{-2x}}{(1 + e^{-x})^2}$.
10. Now, we can substitute $\sigma(x) = \frac{1}{1 + e^{-x}}$ back into this expression to get $\frac{d}{dx}\sigma(x) = \sigma(x)(1 − \sigma(x))$.
