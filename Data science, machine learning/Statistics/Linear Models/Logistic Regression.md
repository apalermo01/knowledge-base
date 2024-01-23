
## Assumptions

### Independent observations

### No multicollinearity among explanatory variables (Use variance inflation factor)

### No extreme outliers (calculate Cook's distance)
- [[Outlier Detection Methods#Cook's Distance]]

### Linear relationship between explanatory variables and logit of response variable (Box-Tidwell test)

**How to check for it**
- "Add log-transformed interaction terms between the continuous independent variables and their natural log into the model"
	- 1) Filter the dataset to keep the continuous independent variables
	- 2) For every continuous independent variable $x$, add $x * \log{x}$ as a new term
	- 3) Run logistic regression and get the p-values of $x*\log x$.
		- If the p-value is not significant: then $x$ is linearly related to the outcome variable
		- if the p-value is significant, then there's a non-linear relationship between the independent variable and the logit
**How to fix it**
- incorporate higher-order terms

### Large sample size (rule of thumb: 10 cases with the least frequent outcome of each explanatory variable)

# References
- https://www.statology.org/assumptions-of-logistic-regression/
- https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290