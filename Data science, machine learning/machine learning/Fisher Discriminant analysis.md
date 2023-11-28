# Fisher Discriminant Analysis

- instead of finding weights that minimize mean squared error, try separating the data based on the distributions instead of adapting weights.

If the vector is $x = (x_1, x_2, .., x_p)^T$ with $\text{class}(x) \in {0, 1}$, then start by projecting the input vector onto one dimension:

$$
z = w^T x
$$

assumption: $\sigma_0 = \sigma_1 = \sigma$ (i.e. the variances in both classes are the same)

The goal is to find $w$ that best separates the data (i.e. the means of the two groups are far apart compared to their spread)



**Fisher's criterion**

$$
J(w) = \frac{(m_1 - m_2)^2}{s_1^2 + s_2^2} = \frac{\bold{w}^T S_b \bold{w}}{\bold{w}^T S_w \bold{w}}
$$

top: distance between projected means<br>
bottom: within-class variance

$$
S_b = (\bold{m_2} - \bold{m_1}) (\bold{m_2} - \bold{m_1})^T
$$

$$
S_w = \sum_{n \in C_1} (x_n - \bold{m_1}) (x_n - \bold{m_1})^T + \sum_{n \in C_2} (x_n - \bold{m_2})(x_n - \bold{m_2})^T
$$

Once you take $\frac{\partial J(w)}{\partial w}$ and set it to 0, you find that:

$$
w \propto S_w^{-1} (\bold{m_2} - \bold{m_1})
$$
## References

- https://towardsdatascience.com/fishers-linear-discriminant-intuitively-explained-52a1ba79e1bb<br>
