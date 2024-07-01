# Derivation

- The objective of logistic regression is to use linear regression to predict a binary outcome. We start by running the output of linear regression through a sigmoid:
$$
P(Y=1|X) = \frac{1}{1+e^{-z}} = \sigma(z)
$$
where
$$
z = \theta_0 + \sum \theta_j x_j = \theta^Tx
$$

The goal is to find the correct values of $\theta$ that maximize $P(Y=1|X)$ and $P(Y=0|X)$ for the correct values of Y. Start by writing the probability of a single datapoint:

$$
P(Y=y | X=x) = \sigma(z)^y \cdot [1-\sigma(z)]^{(1-y)}
$$
Multiply the probability of each datapoint to get the likelihood function

$$
L(\theta) = \prod_{\text{all data}} P(Y=y^{(i)} | X = x^{(i)})  = \prod \sigma(z)^{y^{(i)}} \cdot [1 - \sigma(z)]^{(1-y^{(i)})}
$$

Take the log to get the log-likelihood:
$$
LL(\theta) = \sum y^{(i)} \log(\sigma(z)) + (1-y^{(i)}) \log(1 - \sigma(z))
$$

Now, take the derivative:
$$
\begin{split}
\frac{\partial LL(\theta)}{\partial \theta} &= \sum \frac{y^{(i)}}{\sigma(z)} \frac{\partial \sigma(z)}{\partial z} \frac {\partial z}{\partial \theta} + \frac{1 - y^{(i)}}{1 - \sigma(z)} \frac{- \partial (\sigma(z))}{\partial z} \frac{\partial z}{\partial \theta} \\

&=\sum \biggl[ \frac{y^{(i)}}{\sigma(z)} - \frac{1 - y^{(i)}}{1 - \sigma(z)} \biggr] \frac{\partial \sigma(z)}{\partial z} \frac{\partial z}{\partial \theta} \\
&= \sum \biggl[ \frac{y^{(i)} ( 1 - \sigma(z)) - (1-y^{(i)})\sigma(z)}{\sigma(z) [1 - \sigma(z)]} \biggr] \sigma(z) [1-\sigma(z)] \frac{\partial }{\partial \theta} \theta^T x^{(i)} \\

&= \sum [y^{(i)}(1-\sigma(z)) - (1- y^{(i)})\sigma(z)] x^{(i)} \\

&= \sum [y^{(i)} - y^{(i)}\sigma(z) - \sigma(z) + y^{(i)}\sigma(z)] x^{(i)} \\

&= \sum [y^{(i)} - \sigma(\theta^Tx^{(i)})]x^{(i)}



\end{split}
$$


Therefore for each parameter $\theta_j$:
$$
\frac{\partial LL(\theta)}{\partial \theta_j} = \sum [y^{(i)} - \sigma(\theta^T x^{(i)})] x^{(i)}_j
$$
Now use gradient ascent to update the weights iteratively:

$$
\theta_j^{\text{new}} = \theta_j^{\text{old}} + \eta \cdot \frac{\partial LL(\theta^{\text{old}})}{\partial \theta^{(\text{old})}_j}
$$
# Assumptions

##### Independent observations
- how to identify:
	- plot residuals against the order of observations and check that the pattern is random
- How to fix:
	- Similar to [[Linear Regression Derivation and Assumptions#Independence (no auto-correlation)|Independence assumptions for linear regression]], think of ways to correct for the trend or add seasonality.

##### No multicollinearity among explanatory variables
- How to identify:
	- Cramer's V - statistical test based on the chi-squared statistic that measures the relationship between two ordinal variables (usually going between 0 and 1)
	- VIF (note: I'm not sure that VIF is actually applicable here - definitely shouldn't be used in isolation)
- How to fix:
	- feature engineering: remove one of the highly correlated features
##### No extreme outliers (calculate Cook's distance)
- How to identify:
	- Can use [[Outlier Detection Methods#Cook's Distance|Cook's distance]] to identify the most influential data points
- How to fix:
	- remove them
	- replace with another value like mean or median
	- report those outliers when presenting the model

##### Linear relationship between explanatory variables and logit of response variable

- How identify:
	- "Add log-transformed interaction terms between the continuous independent variables and their natural log into the model"
		- 1) Filter the dataset to keep the continuous independent variables
		- 2) For every continuous independent variable $x$, add $x * \log{x}$ as a new term
		- 3) Run logistic regression and get the p-values of $x*\log x$.
			- If the p-value is not significant: then $x$ is linearly related to the outcome variable
			- if the p-value is significant, then there's a non-linear relationship between the independent variable and the logit
- How to fix:
	- incorporate higher-order terms

##### Large sample size
- Rule of thumb: 10 cases with the least frequent outcome of each explanatory variable

# References
- https://www.statology.org/assumptions-of-logistic-regression/
- https://towardsdatascience.com/assumptions-of-logistic-regression-clearly-explained-44d85a22b290
- https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
- https://stats.stackexchange.com/questions/264628/identifying-multicollinearity-of-categorical-variables-in-a-logistic-regression
- https://stackoverflow.com/questions/35998395/multicollinearity-of-categorical-variables
- https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/pdfs/40%20LogisticRegression.pdf
- https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/pdfs/40%20LogisticRegression.pdf