# Derivation


Linear regression model:

$$
y = \omega_1 x_1 + \omega_2 x_2 + ...
$$
Goal is to minimize mean squared error:

$$
\text{MSE} = \frac{1}{N} \sum (y_{true} - y)^2
$$

We can write the linear model in the form of a matrix equation:

$$
\begin{align}
Y &= X\omega \\

\begin{bmatrix}
y_1 \\  y_2 \\ \vdots \\ y_n
\end{bmatrix}
 &= 

\begin{bmatrix}
x_{11} & \dots & {x_{1n}} \\
\vdots & \ddots & \vdots\\
x_{k1} & \dots & x_{kn}
\end{bmatrix}
\begin{bmatrix}
\omega_1 \\ \omega_2 \\ \vdots \\ \omega_k
\end{bmatrix}

\end{align}
$$
representing a dataset with $k$ variables and $n$ rows. 


The loss term (squared error) then becomes:

$$
L = || Y - X\omega||^2
$$
(remember: Y are the true values, X is the dataset of features, $\omega$ are the weights to be learned)

This can be re-written as:

$$
L = (Y - X\omega) ^T (Y - X\omega)
$$

Now optimize by taking the derivative and setting it equal to zero:

$$
\begin{align}
\frac{\partial L}{\partial \omega} &= 2 (Y-X\omega)^T \frac{\partial (Y- X\omega)}{\partial \omega} \\

&=  2(Y-X\omega)^T (-X) \\
&= 0

\end{align}
$$
Divide both sides by -2
$$
(Y-X\omega)^T X = 0
$$
Take the transpose of both sides and solve for $\omega$
$$
\begin{align}
[(Y-X\omega)^T X]^T &= 0 \\
X^T(Y-X\omega) &= 0 \\
X^T Y - X^T X \omega &= 0 \\
X^T Y &= X^TX \omega \\
(X^T X)^{-1} X^T Y &= \omega
\end{align}
$$
## Some useful matrix identities

$$
\begin{align}
\frac{\partial (u^T A\nu)}{\partial x}
 &= u^T A \frac{\partial \nu}{\partial x} + \nu^T A \frac{\partial u}{\partial x} \\

\frac{\partial u^T u}{\partial x} &= 2 u^T \frac{\partial u}{\partial x}
 
 \end{align}

$$

$$
(AB)^T = B^T A^T
$$

# Assumptions
##### Linearity
- How to identify:
	- plot of observed vs. predicted values
	- plot of residuals vs predicted values (should be symmetrical)
	- watch for a 'bowed' pattern
- How to fix:
	- nonlinear transformations
	- polynomial regression
	- use another model

##### Independence (no auto-correlation)
- How to identify:
	- residual time series plot (residual vs row number)
	- table / plot of residual auto-correlations
	- [[Durbin-Watson Test|Durbin-Watson statistic]]
	- plots of residuals vs independent variables or row number. No matter how it's grouped / sorted, "there should be no correlation between consecutive errors no matter how the rows are sorted"

- How to fix:
	- adjust the lag
	- adjust for seasonality

##### no multicollinearity (input features have no relationship)
- How to identify:
	- test using [[Correlation Matrix|correlation matrix]]
	- "Coefficients of variables are not individually significant (i.e. can't be rejected in the t-test), but can jointly explain the variance of the dependent variable with rejection in the F-test and a high $R^2$. 
	- Tolerance: $T = 1 - R_i^2$
		- $R_i$ = unadjusted coefficient of determination for regressing the ith independent variable on the remaining ones 
	- [[Variance inflation factor]] VIF = 1/T
		- VIF > 4 (or T < 0.25) indicates multicollinearity
	- Situations where high VIF can be ignored
		- only exist in control variables but not variables of interest
		- inclusion of products or powers of other variables (e.g. $x$ and $x^2$)
		- dummy variable that represents more than 2 categories
- How to fix:
	- variable selection (remove some of the offending variables)
	- variable transformation
	- [[PCA|PCA]]
- Consequences:
	- model can give different results every time
	- hard to choose list of significant variables
	- coefficient estimates are not stable (e.g. can't tell how the output will change if one predictor is changed by 1 unit)
	- may cause overfitting
  

##### equality of variance (homoscedasticity)
- How to identify:
	- plot of residuals vs predicted values or residuals vs time or residuals vs independent variables
- How to fix:
	- log transformation
	- not accounting for seasonal patterns
	- check linearity / independence assumptions

##### Errors are normally distributed
- How to identify
	- normal probability or [[quantile plot]] (Q-Q plot?).
	- [[shapiro-wilk test]]
	- [[Kolmogorov-smirnov]], [[jarque-bera]], [[anderson darling]] tests
	- may be better to focus on other tests
- How to fix
	- nonlinear transformations
# References
- https://medium.com/towards-data-science/multi-collinearity-in-regression-fe7a2c1467ea
- https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/assumptions-of-linear-regression/
- https://www.analyticsvidhya.com/blog/2016/07/deeper-regression-analysis-assumptions-plots-solutions/
- https://people.duke.edu/~rnau/testing.htm#linearity
- https://corporatefinanceinstitute.com/resources/data-science/variance-inflation-factor-vif/
- https://towardsdatascience.com/ordinary-least-squares-ols-deriving-the-normal-equation-8da168d740c
