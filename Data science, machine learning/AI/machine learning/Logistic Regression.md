

- used for binary classification
- special case of [[Generalized Linear Models]]


fits the function
$$
y = \frac{1}{1 + e^{-(\omega_0 x_0 + \omega_1 x_1 + ...)}} = \frac{1}{1 + e^{-\omega x}}
$$

using the loss function:
$$
L = \frac{-1}{N} \sum_{i=1}^{N} y_i \log(p(y_i)) + (1-y_i)\log(1-p(y_i))
$$

Optimize weights $w$ to minimize loss function using gradient descent. 

## Gradient derivation


**Gradient of the sigmoid function**

$$
\begin{align}
\frac{dy}{d\omega} &= \frac{(1 + e^{-\omega x})(0) - (1)(-xe^{-\omega x})}{(1 + e^{-\omega x})^2} \\


&= \frac{xe^{-\omega x}}{(1 + e^{-\omega x})^2} \\

&= x \frac{1 - 1 + e^{-\omega x}}{(1 + e^{-\omega x})^2} \\

&= x \bigg( 

\frac{1 + e^{-\omega x}}{(1+ e^{-\omega x})^2} - 
\frac{1}{(1 + e^{-\omega x})^2} 
\bigg) \\

&=x \bigg( 

\frac{1}{1 + e^{-\omega x}} -
\frac{1}{(1 + e^{-\omega x})^2}

\bigg) \\

&= x \bigg(

\frac{1}{1 + e^{-\omega x}} 

\bigg(
1 - \frac{1}{1 + e^{-\omega x}}
\bigg)

\bigg) \\

&= xy(x)[1-y(x)]
\end{align}
$$

**gradient of the loss function**

For conciseness, I'll refer to the ground truth as $y_t$ and the predicted value $y_p$

$$
\begin{align}

\frac{\partial L}{\partial \omega} &= \frac{\partial}{\partial \omega} (
-y_t \ln(y_p) - (1 - y_t)\ln(1-y_p)) \\

&= \frac{-y_t}{y_p}\frac{\partial y_p}{\partial \omega} - \frac{1 - y_t}{1 - y_p}\frac{\partial (1-y_p)}{\partial \omega} \\

&= \frac{-y_t}{y_p}\frac{\partial y_p}{\partial \omega} + \frac{1 - y_t}{1 - y_p} \frac{\partial y_p}{\partial \omega} \\

&= \frac{\partial y_p}{\partial \omega} \bigg(
\frac{-y_t}{y_p} + \frac{1-y_t}{1-y_p}
\bigg) \\

&= \frac{\partial y_p}{\partial \omega} \bigg(
\frac{-y_t ( 1-y_p) + (1-y_t)(y_p)}{y_p(1-y_p)}
\bigg) \\

&= xy_p(1-y_p) \bigg(
\frac{-y_t + y_ty_p + y_p - y_t y_p}{y_p ( 1-y_p)}
\bigg) \\

&= x(y_p - y_t)
\end{align}
$$

## References

- https://medium.com/analytics-vidhya/derivative-of-log-loss-function-for-logistic-regression-9b832f025c2d