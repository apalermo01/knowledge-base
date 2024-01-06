## Summary of support vector machines (theory based)

Note: this discussion came out of prep for a series of tutoring sessions focused on the theory behind machine learning

### support vector machines in plain english
**classification** - draw a line that separates the two classes, where the distance between datapoints closest to the line are as far away from the line as possible<br>
**regression** - fit all your data points into a tube while also making the tube as small as possible.<br>

### Primal formulation of support vector classification

- Consider a dataset with some input matrix and 2 output classes. We'd like to find a decision boundary that perfectly separates the two classes, while also maximizing the distance between the two classes.

The loss function for this model is:

$$
(\hat{\boldsymbol w}, \hat{b}) = \text{argmin}_{(w, b) \in \mathbb{R}^(q+1) } \lbrace \frac{1}{n} \sum_{i=1}^{n} \text{max} (1 - y_i (\boldsymbol{w}^T \boldsymbol{w} + b), 0) + \frac{1}{2} \lambda ||w||_2^2 \rbrace
$$

$\text{max} (1-y_i (\boldsymbol{w}^T\boldsymbol{w} + b), 0)$ punishes datapoints that are on the wrong side of the classifier

$\frac{1}{2} \lambda ||w||_2^2$ punishes datapoints that are close to the decision boundary


### the kernel trick


## References
- https://medium.com/aiguys/the-optimization-behind-svm-primal-and-dual-form-5cca1b052f45<br>

