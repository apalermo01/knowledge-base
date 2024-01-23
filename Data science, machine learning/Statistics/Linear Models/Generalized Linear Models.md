Normal linear regression is in the form of: 

$$
y = x^T\beta
$$
In a generalized linear model, $y$ depends on some nonlinear function of $x^T\beta$:
$$
\eta(y) = x^T \beta
$$
A GLM model can be broken into 3 components:
- Random component ($Y$) - this is the proability distribution of the response variable
- Systematic component - linear combination of explnatory variables ($\beta_0 + \beta_1 x_1 + ...$)
- Link function ($\eta$) - nonlinear function describing the link between random and systematic components. 

## Assumptions
- $Y_1, Y_2, ...$ are independent
- $Y$ does not have to be normaly distributed, but usually comes from an exponential family
- no linear relationship between response and explanatory variables, but does assume a linear relationship between transformed response variable ($\eta(y)$) and explanatory variables
- homogeneity of variance does not need to be satisfied
- parameter estimation uses MLE, not OLS


## popular GLMs
- Simple linear regression
	- $\mu_i = \beta_0 + \beta_1 x_{1i}$
- Binary logistic regression 
	- odds of "success" for a binary response variable (this is logistic regression)
	- $\text{log} (\frac{\pi_i}{1-\pi_i}) = \beta_0 + \beta_1 x_{1i}$
- poisson regression
	- Models how mean of a "count" variable $Y$ depends on explanatory variable
	- $\text{log}(\lambda_i) = \beta_0 + \beta_1 x_{1i}$
	
## References 
- https://en.wikipedia.org/wiki/Generalized_linear_model
- https://online.stat.psu.edu/stat504/lesson/6/6.1