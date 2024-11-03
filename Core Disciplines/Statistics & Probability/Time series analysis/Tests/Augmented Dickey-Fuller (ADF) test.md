Tests whether a time series has [[Unit Root]].

Apply the test to this model:

$$
\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 \Delta y_{t-1} + ... + \delta_{p-1} \Delta y_{t - p + 1} + \epsilon_t
$$

- $\alpha$ is a constant
- $\beta$ is the coefficient on time trend
- $p$ is the lag order for the AR model

- $\alpha = 0$ and $\beta = 0$ corresponds to a random walk
- $\beta = 0$ corresponds to a random walk with drift


- The null hypothesis for the test is $\gamma = 0$ and the alternative is $\gamma < 0$. The test statistic is:
$$
\text{DF}_{\tau} = \frac{\hat \gamma}{\text{SE}(\hat \gamma)}
$$

The idea here is that if there is a unit root process then $y_{t-1}$ will "provide no relevant information in predicting the change $y_t$ besides the one obtained in the lagged changes ($\Delta y_{t-k}$)"

To perform the test, use statsmodels:
https://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html

# References and Links
- https://en.wikipedia.org/wiki/Augmented_Dickey%E2%80%93Fuller_test
- https://www.statsmodels.org/dev/examples/notebooks/generated/stationarity_detrending_adf_kpss.html
- https://www.machinelearningplus.com/time-series/augmented-dickey-fuller-test/