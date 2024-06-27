- Use past forecast errors as the predictor:
$$
y_t = c + \epsilon_t + \theta_1\epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}
$$

- where $c$ represents an average change between observations.
- We call this a $\text{MA}(q)$ model - moving average of order $q$. 
- Since we don't actually observe $\epsilon_t$, this isn't a normal regression model

## References & Links
- https://otexts.com/fpp2/MA.html