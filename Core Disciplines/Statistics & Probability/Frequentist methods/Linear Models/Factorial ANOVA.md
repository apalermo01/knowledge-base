
# Part 1: balanced designs, no interactions

- In normal ANOVA, each observation falls into one of several possible groups and we want to test if these groups have different means on some outcome variable.
- Now, **What if there is more than one grouping variable?** (example: gender and drug given).

Going to the drug example from One-way ANOVA, let's say we're testing 3 drugs (placebo, anxifree, joyzepam) and whether or not the participant is in therapy (yes or no). This would be a 3x2 factorial design.

If there are the same number of people in each group, then we have a **balanced** design.

Writing out the means:
![[Pasted image 20230808203329.png]]

the means with a single dot (e.g. $\mu_{3.}$) represent **marginal means**. $\mu_{3.}$ is the marginal mean for joyzepam, interpreted as the average mood gain across all the people who took Joyzepam regardless of whether they were in therapy.

We can run 2 simple hypothesis tests in terms of the *equality of marginal means*

Does drug have an effect on mood gain?

$$
\begin{align}
H_0: &= \mu_{1.} = \mu_{2.} = \mu_{3.} \\
H_1: &= \text{at least one row is different}
\end{align}
$$



Does therapy have an effect on mood gain?
$$
\begin{align}
H_0: &= \mu_{.1} = \mu_{.2} \\
H_1: &= \text{column means are different}
\end{align}
$$

When you plug this test into software, the outputs will look just like you're running a one-way ANOVA twice, however the p-values won't be the same.

If you run one-way ANOVA using just therapy, ignoring drug, then ANOVA will dump all of the drug variability in the residuals, making the data look more noisy than it actually is.
# Part 2: balanced designs with interactions

An interaction between A and B is said to occur whenever the effect of Factor A is *different* depending on which level of Factor B we're talking about. For example, maybe one drug becomes much more effective when administered with therapy. The ANOVA discussed in part 1 does not account for this.

**Basic idea for dealing with interactions**

The logic here is that if there were no interaction effects, then the marginal means ($\mu_{r.}$) are all equal, and if they're all equal, then they must be equal to the grand mean ($\mu_{..}$) too. Therefore we can define the interaction effect of Factor A at level $r$ to be the difference between the marginal mean ($\mu_{r.}$) and the grand mean ($\mu_{..}$):
$$
\alpha_r = \mu_{r.} - \mu_{..}
$$
we can do the same thing with the column means:
$$
\beta_c = \mu_{.c} - \mu_{..}
$$
Note that both $\alpha_r$ and $\beta_c$ must sum to zero.

If there are no interaction terms, then $\alpha_r$ and $\beta_c$ will perfectly describe the group means $\mu_{rc}$:

$$
\mu_{rc} = \mu_{..} + \alpha_r + \beta_c
$$
**this is our null hypothesis**

If there is an interaction effect, then:

$$
\mu_{rc} \neq \mu_{..} + \alpha_r + \beta_c
$$
which is the alternative hypothesis.

another way to write the alternative hypothesis:

$$
\mu_{rc} = \mu_{..} + \alpha_r + \beta_c + (\alpha \beta)_{rc}
$$
where $(\alpha \beta)_{rc}$ is nonzero for at least one group.

sum of squares is a bit tedious to deal with, but when describing it with the formula for R or scipy you can use:

`mood.gain ~ drug + therapy + drug:therapy`

where `drug:therapy` represents an interaction term

As a shorthand (in R):

`mood.gain ~ drug * therapy`

Like other ANOVA tests, this won't tell you which one of the groups is causing the effect, it will just tell you that there is an effect or interaction somewhere. To nail down which one, you'll need post-hoc tests.

**What if there is a significant interaction but no significant main effect?**
- If this happens (and it happens frequently), then you're probably not interested in the main effect anyway. Even if the tests of the main effects are valid, "when there is a significant interaction effect the main effects rarely test interesting hypotheses". "If you have a significant interaction effect, then you know the groups that comprise the marginal mean aren't homogeneous, so why would you care about those marginal means?" (basically, you're just averaging out the interesting stuff)
# References
- Learning statistics with R chapter 16