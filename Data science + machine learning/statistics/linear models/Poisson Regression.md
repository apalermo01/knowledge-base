- useful for modeling counts of things
	- examples:
		- Are number of motorcycle deaths related to a state's helmet laws? 
		- Does the number of employers conducting on-campus interviews during a year differ for public and private colleges? 
- problem with least squares: variance increases with expectation value (since the variance and mean of a poisson distribution are based on the same number). Also, the dependent variable has to be $\ge 0$ , which is not the case for ordinary least squares. 
- To fix this, model the log of the count ($\lambda$), so:
$$
\text{log} (\lambda_i) = \beta_0 + \beta_1 x_i
$$
Assumptions:
- **Poisson response** - response variable is a count per unit of time or space
- **Independence** - observations must be independent
- **Mean = variance** - mean of a poisson r.v. must be equal to its variance
- **Linearity** log of mean rate must be a linear function in x.

## References
- https://bookdown.org/roback/bookdown-BeyondMLR/ch-poissonreg.html