# Covariance matrices explainer

sources:
- https://datascienceplus.com/understanding-the-covariance-matrix/
- https://towardsdatascience.com/5-things-you-should-know-about-covariance-26b12a0516f1

other related topics:
- linear algebra - matrix basics
- statistics - variance and standard deviation

linked ideas:
AI/machine learning/[[Principal Component Analysis.pdf]]


**Definition of variance**
$$
\sigma^2_{x_i} = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})^2
$$
 **definition of covariance**
 $$
 \sigma(x, y) = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(y_i-\bar{y})
$$
The covariance matrix is a matrix whose entries are the covariances of each combined row. For example - in the case of a 2-dimensional dataset of (x,y) pairs, the covariance matrix would be
$$
C = \begin{bmatrix} \sigma(x, x) & \sigma(x, y) \\
					\sigma(y, x) & \sigma(y, y)\end{bmatrix}
$$
diagonal entries are the variances, off-diagonal are covariances

matrix calculation of covariance:
$$
C = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})(X_i - \bar{X})^T
$$
if the matrix X has zero mean, then:
$$
C = \frac{XX^T}{n-1}
$$

### Interpretation
- describes how two variables vary together
- ![[Pasted image 20220620101619.png]]
- ![[Pasted image 20220620101649.png]]
- What's important are the relative values in the covariance matrix, not absolute values. 