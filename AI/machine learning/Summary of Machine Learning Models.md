

# Regression

## Linear Regression

Just like linear regression in stats, find the weights ($\omega_1, \omega_2, ... \omega_n$) that best fit the equation
$$
y = \omega_1 x_1 + \omega_2 x_2 + ... + \omega_n x_n
$$
also formulized as:
$$
y = X\omega
$$

By default, this algorithm will minimize the mean squared error
$$
\text{MSE} = \frac{1}{N} \sum (y_{\text{pred}} - y_{\text{actual}})^2 = \frac{1}{N} \sum (X\omega - y_{\text{actual}})^2
$$


### Regularization methods
These are variants of linear regression that aim to reduce the magnitude of the coefficients $\omega_n$ . This helps to penalize unimportant predictors and prevent the model from fitting noise.

#### Ridge
Penalizes the squared value of the weights (L2 norm), so that the error term becomes:

$$
\text{error} = \Vert X\omega - y\Vert_2^2 + \alpha\Vert\omega\Vert_2^2
$$

#### Lasso
Penalizes the absolute value of the weights (L1 norm), so that the error term becomes
$$
\text{error} = \Vert X\omega - y \Vert_2^2 + \alpha\Vert\omega\Vert
$$

#### ElasticNet
combined ridge and lasso regression (both L1 and L2 norm)
$$
\text{error} = \Vert X\omega - y\Vert_2^2 + \alpha \rho \Vert \omega\Vert_1 + \frac{\alpha(1-\rho)}{2}\Vert \omega\Vert _2^2
$$

# Classifiers

## Naive bayes

According to Bayes' theorem, the probability of a datapoint having class $y$ given input features $X$ is:

$$
P(y \vert X) = \frac{P(X\vert y) P(y)}{P(X)}
$$

Applying the chain rule and re-arranging results in:
$$
P(y\vert x1, \dots, x_n) \propto P(y) \prod_{i=1}^n P(x_i\vert y)
$$
and we want to select the value (class) of $y$ that maximizes this probability 

- fast / easy to implement
- requires that the predictors are independent
- requires continuous features to be normalized


See the dedicated note on Naive Bayes in this folder for more info. 