# Scikit-Learn Metrics (Regression)

Notes on the scoring options in sklearn's metrics library

### explained_variance_score
[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score)<br> 
[wikipedia](https://en.wikipedia.org/wiki/Explained_variation)

- How much does this model account for the variation of a dataset?
- $\hat{y}$ = estimated target output
- $y$ = correct output
- $\operatorname{Var}$ = variance (square of standard deviation)
$$
\operatorname{explained\_variance}(y, \hat{y}) = 1 - \frac{
    \operatorname{Var}\{ y - \hat{y} \}
    }{
    \operatorname{Var}\{ y\}
    }
$$

- best score = 1, lower = worse

### max_error

[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#max-error)

- computes maximum residual error- worst case error between predicted and true

- $\hat{y}$ = estimated target output
- $y$ = correct output

$$
\operatorname{Max Error}(y, \hat{y}) = \max{|y_i - \hat{y}_i|}
$$

### mean_absolute_error

- Expected value of the absolute error loss ($l_1$-norm loss)
- $\hat{y}$ = estimated target output
- $y$ = correct output
- $n$ = number of samples

$$
\operatorname{MAE}(y, \hat{y}) = \frac{1}{n}
\sum_{i=0}^{n-1}{|y_i - \hat{y}_i|}
$$

### mean_squared_error

[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error)

- expected value of the quadratic loss ($l_2$-norm loss)
- $\hat{y}$ = estimated target output
- $y$ = correct output
- $n$ = number of samples

$$
\operatorname{MSE}(y, \hat{y}) = \frac{1}{n}
\sum_{i=0}^{n-1}{(y_i - \hat{y}_i)^2}
$$

### mean_squared_log_error

[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error)

- Expected value of the squared logarithmic quadratic loss

- expected value of the quadratic loss ($l_2$-norm loss)
- $\hat{y}$ = estimated target output
- $y$ = correct output
- $n$ = number of samples

$$
\operatorname{MSLE}(y, \hat{y}) = \frac{1}{n}
\sum_{i=0}^{n-1}{(\ln(1+y_i) - \ln(1+\hat{y}_i))^2}
$$

- best used when targets have exponential growth

### median_absolute_error

[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#median-absolute-error)

- robust to outliers
- takes median of all differences between targets and predictions
- $\hat{y}$ = estimated target output
- $y$ = correct output
- $n$ = number of samples

$$
\operatorname{MedAE}(y, \hat{y}) = \operatorname{median}(
       |y_1 - \hat{y}_1|, ..., |y_n - \hat{y}_n| 
)
$$

### r2_score

[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score)

- Coefficient of determination
- "The proportion of variance (of y) that has been explained by the independent variables in the model
- Optimal values vary from dataset to dataset
$$
R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n}{(y_i - \hat{y}_i)^2}}{\sum_{i=1}^{n}{(y_i - \bar{y})^2}}
$$

where: 
- $\hat{y}_i$ = predicted value of the $i$-th sample, $y_i$ is the corresponding true value for $n$ samples
- $\bar{y} = \frac{1}{n} \sum_{i=1}^{n}{y_i}$
- $\sum_{i=1}^{n}{(y_i - \hat{y}_i)^2} = \sum_{i=1}^{n}{\epsilon_i^2}$

### mean_poisson, gamma, and tweedie deviances

[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-tweedie-deviance)
[wikipedia](https://en.wikipedia.org/wiki/Tweedie_distribution#The_Tweedie_deviance)

- elicits predicted expecation values of targets

- power = 0: mean squared error
- power = 1: mean poisson deviance
- power = 2: mean gamma devaince

- $\hat{y}$ = estimated target output
- $y$ = correct output
- $n$ = number of samples

![image.png](attachment:image.png)


### mean_absolute_percentage_error

[documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-log-error)

- sensitive to relative errors (not changed by global scaling)
- $\hat{y}$ = estimated target output
- $y$ = correct output
- $n$ = number of samples

$$
\operatorname{MAPE}(y, \hat{y}) = \frac{1}{n}
\sum_{i=0}^{n-1}{\frac{
    |y_i - \hat{y}_i|
    }{
    \max(\epsilon, |y_i|)
    }
    }
$$

- $\epsilon$ is some small positive number to safeguard against $y_i = 0$
