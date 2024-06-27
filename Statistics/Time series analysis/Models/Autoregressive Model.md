- In an autoregression model, we forecast the output using a linear combination of previous values of the output variable.
- Write an autoregressive model like this:
$$
y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \epsilon_t
$$
- where $c$ represents an average change between observations.

- We call this an $\text{AR}(p)$ model, autogressive model of order $p$.


## References & Links
- https://otexts.com/fpp2/AR.html