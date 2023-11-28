Linearity
- how to identify:
	- plot of observed vs. predicted values
	- plot of residuals vs predicted values (should be symmetrical)
	- watch for a 'bowed' pattern
- fix:
	- nonlinear transformations
	- polynomial regression
	- use another model

Independence (no auto-correlation)
- how to identify:
	- residual time series plot (residual vs row number)
	- table / plot of residual auto-correlations
	- Durbin-Watson statistic
	- plots of residuals vs independent variables or row number. No matter how it's grouped / sorted, "there should be no correlation between consecutive errors no matter how the rows are sorted"

- how to fix:
	- adjust the lag
	- adjust for seasonality

no multicollinearity (input features have no relationship)
- how to identify:
	- test using correlation matrix
	- Tolerance: $T = 1 - R^2$
	- Variance inflation factor VIF = 1/T
- how to fix:
	- variable selection (remove some of the offending variables)
	- variable transformation
	- PCA
- Consequences:
	- model can give different results every time
	- hard to choose list of significant variables
	- coefficient estimates are not stable (e.g. can't tell how the output will change if one predictor is changed by 1 unit)
	- may cause overfitting
  

equality of variance (homoscedasticity)
- how to identify:
	- plot of residuals vs predicted values or residuals vs time or residuals vs independent variables
- how to fix:
	- log transformation
	- not accounting for seasonal patterns
	- check linearity / independence assumptions

Errors are normally distributed
- how to identify
	- normal probability or quantile plot (Q-Q plot?).
	- shapiro-wilk test
	- Kolmogorov-smirnov, jarque-bera, anderson darling tests
	- may be better to focus on other tests
- how to fix
	- nonlinear transformations
# References
https://medium.com/towards-data-science/multi-collinearity-in-regression-fe7a2c1467ea
https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-linear-regression/
https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/
https://people.duke.edu/~rnau/testing.htm#linearity