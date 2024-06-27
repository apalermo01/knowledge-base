
- Differencing is the act of taking differences between consecutive observations
- This can help stabilize the mean of a time series, similar to how taking the [[Log Transformation|logarithm]] can stabilize the variance of a time series.

- Use an [[ACF Plot]] to identify non-stationary time series.


## Second-order differencing

If differenced data is not stationary, you can difference a second time to take the second order difference:

$$
\begin{aligned}
y_t'' &= y_t' - y_{t-1}' \\
&= (y_t - y_{t-1}) - (y_{t-1} - y_{t-2}) \\
&= y_t - 2y_{t-1} + y_{t-2}
\end{aligned}
$$

## Seasonal differencing

- Instead of subtracting the previous observation, subtract the previous observation from the same season:
$$
y_t' = y_t - y_{t-m}
$$

where $m$ is the number of periods to lag on


You can determine whether a time series is stationary using a [[Unit Root Test]].

# References & Links

- https://otexts.com/fpp2/stationarity.html