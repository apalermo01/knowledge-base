

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

## Decision Tree Regression
Non parametric - learns a series of rules (e.g. if x1 < 2 then ... ) to predict a value.
- simple to interpret, can be visualized
- prone to overfitting
- recommended to use ensemble methods (e.g. random forest)

## K Nearest Neighbors Regressor
- For every datapoint you want to do predictions on, get the K nearest points in the features space. The average of the nearest labels is the prediction
- Note that this is a distance-based model, so the data should be scaled.
- non parametric
- computationally expensive

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

## Decision tree classification
- Non parametric classifier, works in the same way as decision tree regression but the output is a class
- can be biased - **it is recommended to use a balanced dataset (e.g. run something like  SMOTE)** 

## Logistic Regression

Similar to linear regression, but runs the output of the linear function through a sigmoid:

$$
P(y) = \frac{1}{1 + \exp(\beta_0 + \beta_1 X_1 + ... + \beta_k X_k)}
$$
where $y$ is a binary outcome (0 or 1, success or fail, etc.) and $P(y)$ is the probability of $y$

Note that logistic regression is a special case of [[Generalized Linear Models]]

## K Nearest Neighbors Classifier
- For every datapoint you want to do predictions on, get the K nearest points in the features space. The most common label among those nearest points is the prediction.
- Note that this is a distance-based model, so the data should be scaled.
- non parametric
- computationally expensive

# Clustering

## K-Means
- partitions the dataset into k groups
- iterative
- uses [[Expectation-Maximization]]
- algorithm:
	- 1) randomly select K points
	- 2) for each point in the training data, find the nearest point
	- 3) find the average position of each cluster
	- 4) go back to step 2 until the clusters stop moving or move some distance below a threshold
- Since this is a distance-based model, data should be standardized
- It's usually best to run the model multiple times and pick the model with the lowest sum-of-squared distance, since the model can get stuck in a local minimum. 
# References
- https://scikit-learn.org/stable/modules/tree.html#tree
- https://online.stat.psu.edu/stat462/node/207/
- https://towardsdatascience.com/k-nearest-neighbors-94395f445221
- https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a