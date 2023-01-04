# Hypothesis testing

start with some theory about the world, and determine whether or no the data actually support that theory. 

## A menagerie of hypotheses

Example experiment: test whether clairvoyance exists. 


>Each participant sits down at a table, and is shown a card by an experimenter. The card is black on one side and white on the other. The experimenter takes the card away, and places it on a table in an adjacent room. The card is placed black side up or white side up completely at random, with the randomisation occuring only after the experimenter has left the room with the participant. A second experimenter comes in and asks the participant which side of the card is now facing upwards. Each person sees only one card and gives only one answer and is never in contact with someone who actually knows the right answer

ask the question to N people, and X people gave the correct response. Suppose $N=100$ and $X=62$. Is this enough to make the claim that ESP exists?<br> 

**Research hypotheses vs. statistical hypotheses**<br> 
Research hypothesis: substantive testable scientific claim.<br> 
examples of research hypotheses:<br> 
- linstening to music reduces your ability to pay attention to other things
- Intelligence is related to personality
- Intelligence is speed of information processing

Statistical Hypotheses: mathematically precise and corresponding to specific claims about the characteristics of the data generating mechanism<br> 
For this example, we will call $\theta$ the probability that a subject guesses the correct color<br> 
- If ESP doesn't exists, then the participants are just guessing, so the hypothesis is $\theta = 0.5$
- If ESP does exist, people will perform better than chance, so $\theta > 0.5$
- If ESP does exist, but the colors are reversed, then $\theta < 0.5$
- If ESP does exist, but if we don't know whether or not people are seeing the right color, then $\theta \neq 0.5$

Research hypothesis: ESP Exists<br> 
Statistical hypothesis: $\theta \neq 0.5$

If the study is well-designed, then there is a link between the research hypothesis and the statistical hypothesis.<br> 

**Null hypothesis and alternative hypothesis**<br> 
Invent a new statistical hypothesis (the "Null" hypothesis $H_0$) that is the exact opposite of what we want to believe. Focus on that and neglect the thing we're actually interested in (the "alternative" hypothesis $H_1$)<br> 

Goal: show that the null hypothesis is (probably) false.<br> 

The null hypothesis is deemed to be true unless it can be proven false beyond a reasonable doubt<br> 

The statistical tests are designed to favor the null hypothesis, so that if it is actually true, the chance of it being falsely rejected are low. <br> 

## Two types of errors

![image.png](attachment:image.png)

Goal of the test: control the probability of a type I error (keep it below some fixed probability).<br> 
$\alpha$ = **significance level** (sometimes called size) = probability of a type I error<br> 
A hypothesis test has a significance level of $\alpha$ if the type I error rate is no larger than $\alpha$. 

**power** = probability that we reject the null when it's really false = $1-\beta$, where $\beta$ = type II error rate. 

![image-2.png](attachment:image-2.png)

normally there are 3 different levels of hypothesis testing: 0.05, 0.01, 0.001<br> 

"Better to retain 10 false null hypotheses than to reject a single true one"<br> 

## Test statistics and sample distributions

choose a **test statistic** - what we use to determine whether to accept or reject the null<br> 

figure out which values will cause us to reject the null and which would cause us to keep it. <br> 
What would the **sampling distribution of the test statistic** be if the null were true?<br> 

Because of our experimental design, we know that<br> 
$
X \sim \text{Binomial}(\theta, N)
$

![image.png](attachment:image.png)

## Making decisions

Exactly what values of X should we associate with the Null hypothesis? 

**Critical regions and critical values**<br> 
- X should be very big or very small to reject the null
- If the null hypothesis is true, then the sampling distribution of X is Binomial(0.5, N)
- if $\alpha = 0.05$, the critical region must cover 5% of this sampling distribution

![image.png](attachment:image.png)

**Tails** - critical region with the most extreme values<br> 

**Critical values** - since ~5% of the distribution lies outside the range 40-60, we pick 40 and 60 as the critical values for the test. If $X \le 40$ or $X \ge 60$, then we reject the null. 

If the test statistic falls into the critical region, then the test produced a **significant** result.<br> 

**The difference between a one-sided and two-sided tests**

Two sided test<br> 
$
H_0: \theta = 0.5
$
<br>
$
H_1: \theta \neq 0.5
$

One sided test:<br>
$
H_0: \theta \le 0.5
$
<br>
$
H_1: \theta \gt 0.5
$


## The p-value of a test

2 different ways to interpret P value (Fisher and Neyman)

**A softer view of descision making**<br> 
How can we tell if a result is "barely significant" or "highly significant"?<br> 

Neyman:<br> 
"p is defined to be the smallest Type I error rate ($\alpha$) tha tyou have to be willing to tolerate if you want to reject the null hypothesis<br> 

**The probability of extreme data**<br> 
Fisher:<br>
"we can define p-value as the probability that we would have observed a test statistic that is at least as extreme as the one we actually did get"

**A completely wrong interpretation**<br> 
p = probability that the null hypothesis is true.<br> 
Null hypothesis is a frequentist tool, and you can't have a probability of whether its right or not.<br> 
With the Bayesian lens, p value still doesn't correspond to the probability that the null is true.<br> 

## Reporting the results of a hypothesis test

Should you report the exact p-value or say $p < \alpha$?<br> 

p-values are terribly convenient. Hypothetically, you don't have to say anything about accepting or rejecting the null and just report the raw p-value and let your readers make up their minds.<br> 

gives researcher a lot of leeway- hypothetically you can change your mind later on about how much type I error you're willing to accept<br> 

**Two proposed solutions**<br> 

report p value in terms of widely accepted benchmarks, 0.05, 0.01, and 0.001

tiered approach

![image.png](attachment:image.png)

## Running hypothesis tests in practice

This was an example of a binomial test


```R
binom.test(x=62, n=100, p=0.5)
```


    
    	Exact binomial test
    
    data:  62 and 100
    number of successes = 62, number of trials = 100, p-value = 0.02098
    alternative hypothesis: true probability of success is not equal to 0.5
    95 percent confidence interval:
     0.5174607 0.7152325
    sample estimates:
    probability of success 
                      0.62 



## Effect size, sample size, and power


```R

```
