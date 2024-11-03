big question: we have several groups of observations, do these groups differ in terms of some outcome variable of interest? 

As an example, let's say we have a dataset of results for a clinical trial. 18 participants are trying different combinations of 2 kinds of therapy (9 in CBT, 9 none) and 3 kinds of drugs (6 joyzepam, 6 anxifree, 6 placebo). **How can we tell whether an improvement in mood is significant or just a random coincidence?**

**ANOVA hypotheses**

$$
H_0 : \text{it is true that } \mu_p = \mu_A = \mu_J
$$
$$
H_1 : \text{it is NOT true that } \mu_p = \mu_A = \mu_J
$$
### Some formulas

**Sample variance of Y**
$$
\text{Var}(Y) = \frac{1}{N} \sum_{k=1}^{G} \sum_{i=1}^{N_k} (Y_{ik} - \bar Y) ^2
$$
Where:
- $N$ = number of samples (18)
- $G$ = number of groups (3 - one for each drug)
- $N_k$ = number of people in kth group (6 people each)

Now look at sum of squares, just calculate variance but don't divide by N

$$
\text{SS}_{\text{tot}} = \sum_{k=1}^{G} \sum_{i=1}^{N_k} (Y_{ik} - \bar Y) ^2
$$

Now we break up the sum of squares into two different kinds of variation: 
- **between-group SS** = differences between the means of two classes
- **wthin-group SS** = differences from group means within each group
![[Pasted image 20221101084850.png]]

$$
\text{within group SS} = \text{SS}_w = \sum_{k=1}^{G} \sum_{i=1}^{N_k} (Y_{ik} - \bar Y_k)^2
$$
In other words, we're isolating one group, then calculating the squared difference between each $Y_i$ and the mean of that group $\bar Y_k$, and adding that up for all groups
$$
\text{between group SS} = SS_b = \sum_{k=1}^G N_k (\bar Y_k - \bar Y) ^2
$$
Here, take the mean of every group ($\bar Y_k$)/and compare it against the mean of the whole dataset ($\bar Y$)
and now:
$$
\text{SS}_w + \text{SS}_b = \text{SS}_{tot}
$$

What does this mean? *The variability associated with the outcome variable can be split into two parts: variation due to sample means for different groups ($\text{SS}_b$) and the rest of the variation ($\text{SS}_w$)*

What does this mean for ANOVA? If the means were the same (that is, the null hypothesis is True), then all the sample means would be small and $\text{SS}_b$ would be very small. If the alternative hypothesis is True, then the between-groups differences would be larger (i.e. most of the variation in $Y$ can be explained by taking separate groups) and the within-group SS would be smaller. 

## The F-test
to compare $\text{SS}_w$ to $\text{SS}_b$, we need to run an F-test

**Degrees of freedom**
$$
\begin{align}
\text{df}_b &= G-1 \\
\text{df}_w &= N-G
\end{align}
$$
**Remember**: $N$ is the number of samples (18 for the sample dataset desribed above), $G$ is the number of groups (3 drugs being tested)

Convert sum of squares to a mean squares:
$$
\begin{align}
\text{MS}_b &= \frac{\text{SS}_b}{\text{df}_b} \\
\text{MS}_w &= \frac{\text{SS}_w}{\text{df}_w}
\end{align}
$$
Now our F-ratio is:
$$
F = \frac{\text{MS}_b}{\text{MS}_w}
$$

**Summary**
![[Pasted image 20221101091030.png]]

## Assumptions
- Residuals are normally distributed (use QQ plots or Shapiro-Wilk test)
- Homogeneity of variance / homoscedasicity (every group has the same standard deviation)
- independence (knowing one residual tells you nothing about any other residual)

## Connections to linear models

If you run linear regression on categorical / dummy variables, then the F-test for model fit (e.g. are the coefficients significant?) - that is mathematically equivalent to ANOVA.

Example:
let's say we have 3 groups: A, B, C

If we run ANOVA, we're checking if the means of each of these groups are different

In linear regression, we have a model like: $Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \epsilon$ where $(x_1, x_2) = (1, 0)$ for group A, $(0, 1)$ for group B, and $(0, 0)$ for group C. $\beta_1$ and $\beta_2$ represent the difference in means between group A and C, and group B and C. The F-test in regression will check if $\beta_1$ or $\beta_2$ are different from zero, which is exactly what the ANOVA F-test is checking

If we wanted to account for an interaction term, include a new term that's the product of $x_1$ and $x_2$:$Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + + \beta_3 (x_1 \times x_2) \epsilon$ 
## References
- learning statistics with R chapter 14 (see both the textbook and the assocated [summary](obsidian://open?vault=knowledge-base&file=math%2Fstats%2FLearning%20Statistics%20with%20R%2FLearning%20Statistics%20with%20R%20chapter%2014))