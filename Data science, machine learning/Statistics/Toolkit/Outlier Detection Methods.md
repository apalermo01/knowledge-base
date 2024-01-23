
## Cook's Distance
- Used to detect outliers in [[Linear Regression]] and [[Data science, machine learning/Statistics/Linear Models/Logistic Regression|Logistic Regression]]
- Formula:

$$
D_i = (r_i^2 / p*\text{MSE}) * (h_{ii} / (1-h_{ii})^2)
$$
- $r_i$ = i-th residual
- $p$ = number of coefficients in regression model
- $\text{MSE}$ = mean squared error
- $h_{ii}$ = ith leverage value
- Rule of thumb: any point with $D_i$ = 4/N is an outlier

## References

- https://www.statology.org/how-to-identify-influential-data-points-using-cooks-distance/
- https://www.statology.org/assumptions-of-logistic-regression/