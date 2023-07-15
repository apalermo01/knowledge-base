

# Quick reference

## One variable

### Continuous data

- **One Sampled t-test**: Compare the mean of a sample to a known value or theoretical expectation
- **Paired t-test**: Compare the means of the same group at two different times (e.g., before and after a treatment)
- **One-Sample Wilcoxon test**: Non--parametric test for when the data does not meet the normality assumption. Compare the median of a single column of data to a hypothetical medians

### Categorical data

- **Chi-square goodness of fit**: Test whether the observed proportion of categorical data matches an expected proportion
- **Binomial test**: Test whether the probability of success in a binomial experiment is equal to a specific value

## Two variables

### Continuous - continuous

- **Independent two-sample t-test**: Compare the means of two independent groups
- **Paired t-test**: Compare the means of the same group at two different times (e.g., before and after a treatment)
- **Pearson correlation**: test if two continuous variables are correlated
- **Spearman rank correlation**: Non-parametric test to see if two continuous or ordinal variables are monotonically related
- **Mann-Whitney U test**: Non-parametric alternative to the independent two-sampled t-test

### Categorical - Categorical 

- **Chi-square test for independence**: Test the independence of two categorical variables
- **Fisher's exact test**: Similar to the Chi-square test but used when sample sizes are small

### Categorical - Continuous

- **Independent two-sample t-test**: compare the means of a continuous variable for two categories
- **ANOVA (Analysis of Variance)**: compare the means of a continuous variable for more than two categories
- **Mann-Whitney U test or Kruskal-Wallis Test**: Non-parametric alternatives for the two-sample t-test and ANOVA, respectively

## More than two variables

- **ANOVA (Analysis of variance)**: Test if the means of a continuous variable are different for different categories (more than two) of a categorical variable
- **Multiple Regression**: Test the effect of multiple continuous predictors on a continuous outcome
- **Logistic Regression**: Test the effect of multiple continuous or categorical predictors on a binary outcome
- **Multivariate ANOVA (MANOVA)**: An extension of ANOVA that covers situations where there is more than one dependent variable to be tested


# Details, examples, assumptions, and caveats

### Chi-Square test for independence

- Example: Gather a bunch of robots and a bunch of humans and ask them what they prefer: flowers, puppies, or a properly formatted data file. Do robots and humans have the same preferences?
	- Generating the crosstab gives us this dataset:
	![[Pasted image 20230715150910.png]]
	- $O$ = count of the number of respondants meting that condition
	- $C$ = column totals 
	- $R$ = row totals
	- $N$ = sample size
	- Null hypothesis: All of the following statements are true:
		- $P_{\text{robot}, \text{puppy}} = P_{\text{human}, \text{puppy}}$
		-  $P_{\text{robot}, \text{flower}} = P_{\text{human}, \text{flower}}$
		-  $P_{\text{robot}, \text{data file}} = P_{\text{human}, \text{data file}}$
- test statistic:
$$
\chi^2 = \sum_{i=1}^r\sum_{j=1}^c \frac{(E_{ij} - O_{ij})^2}{E_{ij}}
$$
- degrees of freedom: $df = (r-1)(c-1)$

#### Assumptions of the Chi-Squared test for independence
- Expected frequencies are sufficiently large -> normally we want $N > 5$
- observations are independent
- **if the independence assumption is violated** -> try looking into the McNemar test or Cochran test


### Chi-Square test for goodness of fit

- Example: Ask people to draw 2 cards from a standard deck at random. Were the cards really drawn at random?
	- Null hypothesis: all four suits are chosen with equal probability (e.g. P = (0.25, 0.25, 0.25, 0.25))
	- Alternative hypothesis: At least one of the suit-choice probabilities ISN'T 0.25
	- Test statistic: Compare expected number of observations in each category ($E_i$) with the observed number of observations ($O_i$)
	- Derive the chi-squared statistic using:
	$$
\chi^2 = \sum_{i=1}^{k} \frac{(O_i-E_i)^2}{E_i}
$$
	- Since $O_i$ and $E_i$ represent the probability / frequency of success, they come from a binomial distribution. When $\text{number of samples} \times \text{probability of success}$ is large enough, this becomes a normal distribution. And squaring things that come from normal distributions and adding them up gives you a chi-squared distribution
- degrees of freedom
	- main idea: calculate DoF by counting the number of distinct quantities used to describe the data and subtract off all the constraints.
	- For this case, we describe the data using 4 numbers corresponding to the observed frequencies of each category. There is one fixed constraint: if we know the sample size, we can figure out how many people chose spades given we know how many people picked hearts, clubs, diamonds, etc. Therefore our degrees of freedom are $N-1$ for $N$ variables plus the constraint that the sum of the probabilities must sum to 1.
- This is always a 1-sided test

#### Assumptions of the Chi-Squared goodness-of-fit test
- Expected frequencies are sufficiently large -> normally we want $N > 5$
- observations are independent
- **if the independence assumption is violated** -> try looking into the McNemar test or Cochran test


### Correction to chi-squared tests when there's 1 DoF

This is called the **Yates Correction** or **continuity correction**. Basically, the chi-squared test is based on the assumption that the binomial distributions look like normal distributions with large $N$. When there's 1 DoF (i.e, a 2x2 contingency table) and $N$ is small, the test statistic is generally too big. **Yates** proposed this correction as more of a hack, probably not derived from anything and just based on empirical evidence:

$$
\chi^2 = \sum_i \frac{j(|E_i - O_i| - 0.5)^2}{E_i}
$$



### Fisher's exact test

- Use this when you don't have enough samples to do a chi-squared test
- Start with the same contingency table
![[Pasted image 20230715152646.png]]
- Calculate the probability that we would have obtained the observed frequencies that we did ($O_{11}, O_{12}, O_{21}, O_{22}$) given the row and column totals:

$$
P(O_{11}, O_{12}, O_{21}, O_{22} | R_1, R_2, C_1, C_2)
$$
- This is describe by a hypergeometric distribution
# References

- Multiple conversations with chatGPT 
- Learning Statistics with R, chapter 12.1, 12.7