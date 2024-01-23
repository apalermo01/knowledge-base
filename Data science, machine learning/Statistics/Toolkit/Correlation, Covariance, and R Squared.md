
**Covariance**: a measure of the joint variation between two variables
$$
\text{Cov}(X, Y) = \frac{1}{N-1} \sum (X_i - \bar X)(Y_i - \bar Y)
$$
**Correlation (aka pearson correlation coefficient)**: standardize the covariance
$$
R = \frac{\text{Cov}(X, Y)}{\sigma_x \sigma_y}
$$

- Correlation ranges between -1 and 1. 0 is no relationship, -1 perfect negative relationship, and 1 is a perfect positive relationship.
- See Anscombe's quartet for an example on the shortcomings of using just correlation
- ALWAYS PLOT THE DATA BEFORE MAKING AN INTERPRETATION

**Spearman's rank correlation**
- If we want to capture the fact that there's an ordinal relationship between two variables, not how well they form a line, then convert the data to rank (i.e. instead of 0.1, 0.5, 0.2, 1, 0.9, do 1, 3, 2, 5, 4) then calculate the pearson correlation coefficient.

**R^2 (aka coefficient of determination)**
- This is the square of pearson's correlation coefficient. 
- When applied to linear [[One-way ANOVA|regression models]], it is interpreted as the proportion of the variance in the outcome variable that can be explained by the predictor

$$
R^2 = 1 - \frac{\text{sum of squares residuals}}{\text{sum of squares total}}
$$
- So an $R^2$ of 0.8 means the predictor explains 80% of the variance in the outcome


**Adjusted R^2**
- This is $R^2$ after taking degrees of freedom into account
$$
\text{adj. } R^2 = 1 - (\frac{\text{SS}_{res}}{\text{SS}_{tot}} \times \frac{N-1}{N-K-1})
$$
Where K is the number of predictors and N is the number of observations


And of course, keep in mind **CORRELATION DOES NOT IMPLY CAUSATION**

## References
- Learning Statistics with R section 15.4
- Learning Statistics with R section 5.7

