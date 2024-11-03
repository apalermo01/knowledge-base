- In an autoregression model, we forecast the output using a linear combination of previous values of the output variable.
- Write an autoregressive model like this:
$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \epsilon_t
$$
- where $c$ represents an average change between observations.

- We call this an $\text{AR}(p)$ model, autogressive model of order $p$.


## Finding the most optimal order
- One way to find the optimal $p$ for an AR model is to use the [[Autocorrelation function (ACF)#Partial autocorrelation function |PACF plot]]. The partial autocorrelations that are significantly different from zero are usually ones that you want to include in the order, so try using the largest value of the PACF plot that is significant.
- You can also iteratively fit the AR model on different orders and find the one that minimizes error.

## References & Links
- https://otexts.com/fpp2/AR.html
- https://online.stat.psu.edu/stat501/book/export/html/996
- https://stats.stackexchange.com/questions/47780/how-to-decide-the-optimal-ar-model-order