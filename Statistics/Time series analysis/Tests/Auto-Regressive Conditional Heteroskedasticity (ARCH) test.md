ARCH is a model / test used to check for Heteroskedasticity in [[Time Series - Header|time series]] models. In other words, it measures whether the errors (residuals) in a time-series model depend on time. This is a check for one of the [[Linear Regression Derivation and Assumptions#equality of variance (homoscedasticity)|key assumptions of linear regression]]. In terms of real-world applications, this could be described as clustering volatility (e.g. prices in a market going through periods of stability and instability).

To perform the test, first fit an autoregressive model and obtain the error terms ($\epsilon_t$).
$$
\text{AR}(q) y_t = \alpha_0 + \alpha_1 y_{t-1} + ... + \alpha_q y_{t-q} + \epsilon_t = \alpha_0 + \sum_{i=1}^q \alpha_i y_{t-i} + \epsilon_t
$$

Break the residuals into 2 components - a white noise component ($z_t$) and a time-dependent component ($\sigma_t$). 

$$
\epsilon_t = \sigma_t z_t
$$
Model the time-dependent variance as a linear regression model where the input features are all the previous residuals (squared so we're dealing with variances).

$$
\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + ... + \alpha_q \epsilon_{t-q}^2 = \alpha_0 + 
$$
In practice, take the set of residuals from the AR model and build another AR model out of the square of those residuals (note that none of the resources I've found explicitly say to build an AR model out of the residuals, but functionally it looks the same):

$$
\hat \epsilon_t^2 = \alpha_0 + \sum_{i=1}^q \alpha_i \hat \epsilon_{t-i}^2
$$
If there is truly no heteroskedasticity (the null hypothesis), then the current value of the residual will not depend on the previous values, so $\alpha_i = 0$ for $i > 0$. If we're working with $T$ residuals, then the test statistic is $T'R^2$, where $R^2$ is the [[Correlation, Covariance, and R Squared#$R 2$ (aka coefficient of determination)|coefficient of determination]]) and $T' = T - q$, which follows a [[Chi-squared]] distribution with $q$ degrees of freedom. 

# References & Links
- https://www.investopedia.com/terms/a/autoregressive-conditional-heteroskedasticity.asp
- https://en.wikipedia.org/wiki/Autoregressive_conditional_heteroskedasticity