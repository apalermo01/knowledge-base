- Use past forecast errors as the predictor:
$$
y_t = c + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}
$$

- where $c$ represents an average change between observations.
- We call this a $\text{MA}(q)$ model - moving average of order $q$. 
- Since we don't actually observe $\epsilon_t$, this isn't a normal regression model

## Invertible moving average model

It is possible to write a $\text{MA(1)}$ as an $\text{AR}(\infty)$ model. Details are in the references.

## Finding the most optimal order
- look at the [[Autocorrelation function (ACF)|ACF Plot]] - the most optimal $q$ is the largest significant lag. 
## References & Links
- https://otexts.com/fpp2/MA.html
- https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method#Identify_p_and_q