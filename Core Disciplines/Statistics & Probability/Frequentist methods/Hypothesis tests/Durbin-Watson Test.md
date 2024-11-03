 The Durbin-Watson test is a test for auto-correlation in [[Linear Regression Derivation and Assumptions#Independence (no auto-correlation)|linear regression]] - in other words, it verifies that a given observation does not depend on the other observations. 
 It ranges from 0 to 4, where 2 indicates no autocorrelation. < 2 means there is a positive autocorrelation, >2 means there is a negative autocorrelation. 
 For a rule of thumb, values between 1.5 and 2.5 generally mean no autocorrelation. 

The test statistic is given by:

$$
d = \frac{\sum_{t=2}^T (e_t - e_{t-1}^2)}{\sum_{t=1}^T e_t^2}
$$

where $e_t$ are residuals. 

Note - this test is not applicable when lagged dependent variables are in the feature matrix (see wikipedia article below).

## Links / References
- https://www.investopedia.com/terms/d/durbin-watson-statistic.asp
- https://en.wikipedia.org/wiki/Durbin%E2%80%93Watson_statistic