# Comparing 2 means

## The one-sample z-test

**The inference problem that the test addresses**

Get the average and sd of grades in a stats course. Do the psychology students in the course (20 out of several hundred) tend to score the same, higher, or lower? 

We know for the whole class, avg = 67.5, sd = 9.5<br> 

load the data for just the psych students


```R
load("zeppo.Rdata")
print(grades)
```

     [1] 50 60 60 64 66 66 67 69 70 74 76 76 77 79 79 79 81 82 82 89



```R
mean(grades)
```


72.3


We know $\mu = 67.5$ (population average) and $\bar{X} = 72.3$ (sample average)<br> 

assume that the psychology students have the same standard deviation as the rest of the class, which means the population standard deviation is $\sigma=9.5$. Since the students are graded on a curve, the grades come from a normal distribution<br> 

Research hypothesis: Is the population mean $\mu$ for psychology students the same as the popualtion mean for the rest of the class? 

**Constructing the hypothesis test**

$$
H_0: \mu = 67.5
$$
$$
H_1: \mu \neq 67.5
$$

two important pieces of information:<br> 
- the psychology students are normally distributed
- the true standard-deviation of these scores $\sigma$ is known to be 9.5 (this is always an assumption)

![image.png](attachment:image.png)

What is the test statistic?<br> 
start with $\bar{X} - \mu_0$

We know that $\X \sim \text{Normal}(\mu_0, \sigma^2$<br>

The sampling distirbution of the mean $\bar{X}$ is also normal with a mean $\mu$, but standard deviation of the sampling distribution (standard error of the mean) is:
$$
\text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{N}}
$$

So if the null hypothesis is true, then:<br> 
$$
\bar{X} \sim \text{Normal}(\mu_0, \text{SE}(\bar{X}))
$$

convert sample mean $\bar X$ into a standard score. 

$$
z_{\bar X} = \frac{\bar X - \mu_0}{\text{SE}(\bar X)} = \frac{\bar X - \mu_0}{\sigma / \sqrt N}
$$

![image-2.png](attachment:image-2.png)

**A worked example using R**


```R
# calculate sample mean
sample.mean <- mean(grades)
paste("sample mean = ", sample.mean)

# known population params
mu.null <- 67.5
sd.true <- 9.5

# sample size
N <- length(grades)
paste("sample size = ", N)

# calculate standard error of the mean
sem.true <- sd.true / sqrt(N)
paste("standard error of the mean = ", sem.true)

# calculate z score
z.score <- (sample.mean - mu.null) / sem.true
paste("z score = ", z.score)
```


<span style=white-space:pre-wrap>'sample mean =  72.3'</span>



<span style=white-space:pre-wrap>'sample size =  20'</span>



<span style=white-space:pre-wrap>'standard error of the mean =  2.1242645786248'</span>



<span style=white-space:pre-wrap>'z score =  2.25960553515768'</span>


For a 2-sided test, 2.26>2.57, so we can say $p < 0.05, p > 0.01$

to get exact p-value


```R
upper.area <- pnorm(q=z.score, lower.tail=FALSE)
upper.area
```


0.0119228718824699



```R
lower.area <- pnorm(q=-z.score, lower.tail=TRUE)
lower.area
```


0.0119228718824699



```R
# get p-value
p.value <- lower.area + upper.area
p.value
```


0.0238457437649398


**Assumptions of the z-test**<br> 
- normality
- independence
- known standard deviation -> this is kind of stupid for real-world data. This assumption is always wrong but sometimes we can get away with it

## The one-sample t-test

**Introducing the t-test**

we can't say that the population sd is 9.5, but we CAN say that the sample sd ($\hat \sigma$) is:


```R
sd(grades)
```


9.52061475237591


we have to accoutn for the fact that this sd is an estimate. 

Here, we use a t-statistic, which is just like the z-score except we're using the sample sd

$$
t = \frac{\bar X - \mu}{\hat \sigma / \sqrt N}
$$

the sampling distribution is now a t-distribution with $N-1$ degrees of freedom.<br> 

It's much like the normal distribution but with heavier tails. 

For large N, t-test and z-test converge since estimate of population sd become much more precise

**Doing the test in R**


```R
t.test(grades, mu=67.5)
```


    
    	One Sample t-test
    
    data:  grades
    t = 2.2547, df = 19, p-value = 0.03615
    alternative hypothesis: true mean is not equal to 67.5
    95 percent confidence interval:
     67.84422 76.75578
    sample estimates:
    mean of x 
         72.3 



with lsr package


```R
library(lsr)
oneSampleTTest(x=grades, mu=67.5)
```


    
       One sample t-test 
    
    Data variable:   grades 
    
    Descriptive statistics: 
                grades
       mean     72.300
       std dev.  9.521
    
    Hypotheses: 
       null:        population mean equals 67.5 
       alternative: population mean not equal to 67.5 
    
    Test results: 
       t-statistic:  2.255 
       degrees of freedom:  19 
       p-value:  0.036 
    
    Other information: 
       two-sided 95% confidence interval:  [67.844, 76.756] 
       estimated effect size (Cohen's d):  0.504 



**Assumptions of the 1-sample t-test**

- Normality
- Independence

## The independent samples t-test (student)

- two different groups of observations (i.e. two different groups of participants, each one being exposed to a different condition)


**The data**


```R
load("harpo.Rdata")
who(TRUE)
```


       -- Name --    -- Class --   -- Size --
       grades        numeric       20        
       harpo         data.frame    33 x 2    
        $grade       numeric       33        
        $tutor       factor        33        
       lower.area    numeric       1         
       mu.null       numeric       1         
       N             integer       1         
       p.value       numeric       1         
       sample.mean   numeric       1         
       sd.true       numeric       1         
       sem.true      numeric       1         
       upper.area    numeric       1         
       z.score       numeric       1         



```R
head(harpo)
```


<table>
<thead><tr><th scope=col>grade</th><th scope=col>tutor</th></tr></thead>
<tbody>
	<tr><td>65        </td><td>Anastasia </td></tr>
	<tr><td>72        </td><td>Bernadette</td></tr>
	<tr><td>66        </td><td>Bernadette</td></tr>
	<tr><td>74        </td><td>Anastasia </td></tr>
	<tr><td>73        </td><td>Anastasia </td></tr>
	<tr><td>71        </td><td>Bernadette</td></tr>
</tbody>
</table>



![image.png](attachment:image.png)


**Introducing the test**

goal: determine whether 2 independent samples of data are drawn from populations of the same mean (null) or different mean (alternative)

$$
H_0: \mu_1 = \mu_2
$$

$$
H_1: \mu_1 \neq \mu_2
$$


test statistic:<br> 

$$
t = \frac{\bar X_1 - \bar X_2}{\text{SE}}
$$

**A "pooled estimate" of the standard deviation**<br> 

for the Student t-test, we assume that the two groups have the same population standard deviation. <br> 

to calculate this , use a weighted average of the variance<br> 

weights:
$$
w_1 = N_1 - 1
$$

$$
w_2 = N_2 - 1
$$

pooled estimate of variance:<br> 

$$
\hat \sigma_p^2 = \frac{w_1 \hat \sigma_1^2 + w_2 \hat \sigma_2^2}{w_1 + w_2}
$$

**The same pooled estimate, described differently**

$$
\hat \sigma_p^2 = \frac{\sum_{ik} (X_{ik} - \bar X_k)^2}{N-2}
$$

where $k$ = group and $i$ = student

**Completing the test**

to get standard error, we want standard error of the difference. If they really have the same sd, then:<br> 

$$
\text{SE}(\bar X_1 - \bar X_2) = \hat \sigma \sqrt{\frac{1}{N_1} + \frac{1}{N_2}}
$$

so the t-statistic is:



**doing the test in R**

use function in lsr package


```R
head(harpo)
```


<table>
<thead><tr><th scope=col>grade</th><th scope=col>tutor</th></tr></thead>
<tbody>
	<tr><td>65        </td><td>Anastasia </td></tr>
	<tr><td>72        </td><td>Bernadette</td></tr>
	<tr><td>66        </td><td>Bernadette</td></tr>
	<tr><td>74        </td><td>Anastasia </td></tr>
	<tr><td>73        </td><td>Anastasia </td></tr>
	<tr><td>71        </td><td>Bernadette</td></tr>
</tbody>
</table>




```R
independentSamplesTTest(
  formula=grade~tutor, # formula specifying outcome and group variables
  data=harpo,          # dataframe that contains the variables
  var.equal=TRUE       # assume that the two groups have the same variance
)
```


    
       Student's independent samples t-test 
    
    Outcome variable:   grade 
    Grouping variable:  tutor 
    
    Descriptive statistics: 
                Anastasia Bernadette
       mean        74.533     69.056
       std dev.     8.999      5.775
    
    Hypotheses: 
       null:        population means equal for both groups
       alternative: different population means in each group
    
    Test results: 
       t-statistic:  2.115 
       degrees of freedom:  31 
       p-value:  0.043 
    
    Other information: 
       two-sided 95% confidence interval:  [0.197, 10.759] 
       estimated effect size (Cohen's d):  0.74 



important note: this confidence interval is for the difference between the group means

The confidence interval tells you that if you do this experiment over and over again, then the true difference between the means will lie between 0.2 and 10.8 95% of the time

**Positive and negative t-values**

$$
t = \frac{\text{mean 1} - \text{mean 2}}{(\text{SE})}
$$

if mean 2 > mean 1, then t will be negative. 

aim to report the t-statistic in a way that the numbers match up with the text

e.g. if mean 1 was Anastasia, then we should say "Anastasia's class had higher grades than... ". Meanwhile if mean 2 was Bernadette, we should say "Beradette's class had lower grades than... "<br>

**Assumptions of the t-test**

- Normality
- Independence
- homogenity of variance (homoscedasticity) - the population standard deviation is the same in both groups

## The independent samples t-test (Welch)

remove assumption of same variance

standard error becomes: 
$$
\text{SE} (\bar X_1 - \bar X_2) = \sqrt{\frac{\hat \sigma_1^2}{N_1} + \frac{\hat \sigma_2^2}{N_2}}
$$

different calculation for dof:

![image.png](attachment:image.png)

**doing the test in R**


```R
independentSamplesTTest(
  formula = grade~tutor,
  data=harpo
)
```


    
       Welch's independent samples t-test 
    
    Outcome variable:   grade 
    Grouping variable:  tutor 
    
    Descriptive statistics: 
                Anastasia Bernadette
       mean        74.533     69.056
       std dev.     8.999      5.775
    
    Hypotheses: 
       null:        population means equal for both groups
       alternative: different population means in each group
    
    Test results: 
       t-statistic:  2.034 
       degrees of freedom:  23.025 
       p-value:  0.054 
    
    Other information: 
       two-sided 95% confidence interval:  [-0.092, 11.048] 
       estimated effect size (Cohen's d):  0.724 



This test isn't significant now! don't need to worry about interpreting why one test was significant and not the other, what's important is that you carefully consider which test to report. Student test is more powerful (lower type II error rate) than Welch, but if the population variances really are different, then you might end up with a higher type I error rate. 

**Assumptions of the test**

- Normality
- independence

## The paired-samples t-test

What if we have a repeated measures design, where each participant is measured in both conditions?<br> 

**The data**


```R
load("chico.Rdata")
who(TRUE)
```


       -- Name --      -- Class --   -- Size --
       chico           data.frame    20 x 3    
        $id            factor        20        
        $grade_test1   numeric       20        
        $grade_test2   numeric       20        
       grades          numeric       20        
       harpo           data.frame    33 x 2    
        $grade         numeric       33        
        $tutor         factor        33        
       lower.area      numeric       1         
       mu.null         numeric       1         
       N               integer       1         
       p.value         numeric       1         
       sample.mean     numeric       1         
       sd.true         numeric       1         
       sem.true        numeric       1         
       upper.area      numeric       1         
       z.score         numeric       1         



```R
head(chico)
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>grade_test1</th><th scope=col>grade_test2</th></tr></thead>
<tbody>
	<tr><td>student1</td><td>42.9    </td><td>44.6    </td></tr>
	<tr><td>student2</td><td>51.8    </td><td>54.0    </td></tr>
	<tr><td>student3</td><td>71.7    </td><td>72.3    </td></tr>
	<tr><td>student4</td><td>51.6    </td><td>53.4    </td></tr>
	<tr><td>student5</td><td>63.5    </td><td>63.8    </td></tr>
	<tr><td>student6</td><td>58.0    </td><td>59.3    </td></tr>
</tbody>
</table>




```R
library(psych)
describe(chico)
```


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>id*</th><td>1          </td><td>20         </td><td>10.500     </td><td>5.916080   </td><td>10.5       </td><td>10.50000   </td><td>7.41300    </td><td> 1.0       </td><td>20.0       </td><td>19.0       </td><td> 0.00000000</td><td>-1.3809286 </td><td>1.322876   </td></tr>
	<tr><th scope=row>grade_test1</th><td>2          </td><td>20         </td><td>56.980     </td><td>6.616137   </td><td>57.7       </td><td>56.91875   </td><td>7.70952    </td><td>42.9       </td><td>71.7       </td><td>28.8       </td><td> 0.05403574</td><td>-0.3549756 </td><td>1.479413   </td></tr>
	<tr><th scope=row>grade_test2</th><td>3          </td><td>20         </td><td>58.385     </td><td>6.405612   </td><td>59.7       </td><td>58.35000   </td><td>6.44931    </td><td>44.6       </td><td>72.3       </td><td>27.7       </td><td>-0.05385765</td><td>-0.3892926 </td><td>1.432338   </td></tr>
</tbody>
</table>



![image.png](attachment:image.png)

in pannel b, almost all dots are falling above the line, suggesting the near universal improvement. 

make a variable for improvement


```R
chico$improvement <- chico$grade_test2 - chico$grade_test1
head(chico)
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>grade_test1</th><th scope=col>grade_test2</th><th scope=col>improvement</th></tr></thead>
<tbody>
	<tr><td>student1</td><td>42.9    </td><td>44.6    </td><td>1.7     </td></tr>
	<tr><td>student2</td><td>51.8    </td><td>54.0    </td><td>2.2     </td></tr>
	<tr><td>student3</td><td>71.7    </td><td>72.3    </td><td>0.6     </td></tr>
	<tr><td>student4</td><td>51.6    </td><td>53.4    </td><td>1.8     </td></tr>
	<tr><td>student5</td><td>63.5    </td><td>63.8    </td><td>0.3     </td></tr>
	<tr><td>student6</td><td>58.0    </td><td>59.3    </td><td>1.3     </td></tr>
</tbody>
</table>




```R
hist(chico$improvement)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%2013_49_0.png)
    



```R
ciMean(chico$improvement)
```


<table>
<thead><tr><th scope=col>2.5%</th><th scope=col>97.5%</th></tr></thead>
<tbody>
	<tr><td>0.9508686</td><td>1.859131 </td></tr>
</tbody>
</table>



Therefore, we have a real within-student improvement with very large between-student differences

**What is the paired sample t-test?**

We are interested in **Within-subject**, not **between-subject** variability

Run a 1-sample t-test on the improvement
$$
D_i = X_{i1} - X_{i2}
$$

**Doing the test in R, part 1**


```R
oneSampleTTest(chico$improvement, mu=0)
```


    
       One sample t-test 
    
    Data variable:   chico$improvement 
    
    Descriptive statistics: 
                improvement
       mean           1.405
       std dev.       0.970
    
    Hypotheses: 
       null:        population mean equals 0 
       alternative: population mean not equal to 0 
    
    Test results: 
       t-statistic:  6.475 
       degrees of freedom:  19 
       p-value:  <.001 
    
    Other information: 
       two-sided 95% confidence interval:  [0.951, 1.859] 
       estimated effect size (Cohen's d):  1.448 



we can also do this without making any new variables<br> 

can run pairedSamplesTTest() in lsr package. input a 1-sided formula


```R
pairedSamplesTTest(
  formula=~grade_test2+grade_test1,
  data=chico
)
```


    
       Paired samples t-test 
    
    Variables:  grade_test2 , grade_test1 
    
    Descriptive statistics: 
                grade_test2 grade_test1 difference
       mean          58.385      56.980      1.405
       std dev.       6.406       6.616      0.970
    
    Hypotheses: 
       null:        population means equal for both measurements
       alternative: different population means for each measurement
    
    Test results: 
       t-statistic:  6.475 
       degrees of freedom:  19 
       p-value:  <.001 
    
    Other information: 
       two-sided 95% confidence interval:  [0.951, 1.859] 
       estimated effect size (Cohen's d):  1.448 



**Doing the test in R, part 2**

repeated measure tests can be expressed in wide-form or long-form<br> 

data are currently in wide-form (every row is a unique person)

most tools in R want data in long form (paired samples t-test is an exception)


```R
head(chico)
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>grade_test1</th><th scope=col>grade_test2</th><th scope=col>improvement</th></tr></thead>
<tbody>
	<tr><td>student1</td><td>42.9    </td><td>44.6    </td><td>1.7     </td></tr>
	<tr><td>student2</td><td>51.8    </td><td>54.0    </td><td>2.2     </td></tr>
	<tr><td>student3</td><td>71.7    </td><td>72.3    </td><td>0.6     </td></tr>
	<tr><td>student4</td><td>51.6    </td><td>53.4    </td><td>1.8     </td></tr>
	<tr><td>student5</td><td>63.5    </td><td>63.8    </td><td>0.3     </td></tr>
	<tr><td>student6</td><td>58.0    </td><td>59.3    </td><td>1.3     </td></tr>
</tbody>
</table>




```R
chico2 <- wideToLong(chico, within="time")
head(chico2)
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>improvement</th><th scope=col>time</th><th scope=col>grade</th></tr></thead>
<tbody>
	<tr><td>student1</td><td>1.7     </td><td>test1   </td><td>42.9    </td></tr>
	<tr><td>student2</td><td>2.2     </td><td>test1   </td><td>51.8    </td></tr>
	<tr><td>student3</td><td>0.6     </td><td>test1   </td><td>71.7    </td></tr>
	<tr><td>student4</td><td>1.8     </td><td>test1   </td><td>51.6    </td></tr>
	<tr><td>student5</td><td>0.3     </td><td>test1   </td><td>63.5    </td></tr>
	<tr><td>student6</td><td>1.3     </td><td>test1   </td><td>58.0    </td></tr>
</tbody>
</table>




```R
chico2 <- sortFrame(chico2, id)
head(chico2)
```


<table>
<thead><tr><th></th><th scope=col>id</th><th scope=col>improvement</th><th scope=col>time</th><th scope=col>grade</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td>student1 </td><td>1.7      </td><td>test1    </td><td>42.9     </td></tr>
	<tr><th scope=row>21</th><td>student1 </td><td>1.7      </td><td>test2    </td><td>44.6     </td></tr>
	<tr><th scope=row>10</th><td>student10</td><td>1.3      </td><td>test1    </td><td>61.9     </td></tr>
	<tr><th scope=row>30</th><td>student10</td><td>1.3      </td><td>test2    </td><td>63.2     </td></tr>
	<tr><th scope=row>11</th><td>student11</td><td>1.4      </td><td>test1    </td><td>50.4     </td></tr>
	<tr><th scope=row>31</th><td>student11</td><td>1.4      </td><td>test2    </td><td>51.8     </td></tr>
</tbody>
</table>




```R
pairedSamplesTTest(
  formula = grade~time,
  data=chico2,
  id="id"
)
```


    
       Paired samples t-test 
    
    Outcome variable:   grade 
    Grouping variable:  time 
    ID variable:        id 
    
    Descriptive statistics: 
                 test1  test2 difference
       mean     56.980 58.385     -1.405
       std dev.  6.616  6.406      0.970
    
    Hypotheses: 
       null:        population means equal for both measurements
       alternative: different population means for each measurement
    
    Test results: 
       t-statistic:  -6.475 
       degrees of freedom:  19 
       p-value:  <.001 
    
    Other information: 
       two-sided 95% confidence interval:  [-1.859, -0.951] 
       estimated effect size (Cohen's d):  1.448 



In a sense, we're interested in how the outcome variable (grade) is related to the grouping variable (time), after taking into account that there are individual differences between people (id). In a sense, id is a second predictor. This means we can write a formula like `grade ~ time + (id)`. This says the variable on the left is the outcome variable, bracketed term (id) is the id variable, and the other term is the grouping variable (time)


```R
pairedSamplesTTest(grade~time+(id),chico2)
```


    
       Paired samples t-test 
    
    Outcome variable:   grade 
    Grouping variable:  time 
    ID variable:        id 
    
    Descriptive statistics: 
                 test1  test2 difference
       mean     56.980 58.385     -1.405
       std dev.  6.616  6.406      0.970
    
    Hypotheses: 
       null:        population means equal for both measurements
       alternative: different population means for each measurement
    
    Test results: 
       t-statistic:  -6.475 
       degrees of freedom:  19 
       p-value:  <.001 
    
    Other information: 
       two-sided 95% confidence interval:  [-1.859, -0.951] 
       estimated effect size (Cohen's d):  1.448 



## One sided tests

what if we're interested in finding out if the true mean for Dr. Zeppo's class is lower than 67.5%?

use one.sided argument


```R
oneSampleTTest(grades, 67.5, one.sided = "greater")
```


    
       One sample t-test 
    
    Data variable:   grades 
    
    Descriptive statistics: 
                grades
       mean     72.300
       std dev.  9.521
    
    Hypotheses: 
       null:        population mean less than or equal to 67.5 
       alternative: population mean greater than 67.5 
    
    Test results: 
       t-statistic:  2.255 
       degrees of freedom:  19 
       p-value:  0.018 
    
    Other information: 
       one-sided 95% confidence interval:  [68.619, Inf] 
       estimated effect size (Cohen's d):  0.504 



in Dr. Harpo's class, see if Anastasia's students had higher grades than Bernadette's 


```R
independentSamplesTTest(formula=grade~tutor, data=harpo, one.sided="Anastasia")
```


    
       Welch's independent samples t-test 
    
    Outcome variable:   grade 
    Grouping variable:  tutor 
    
    Descriptive statistics: 
                Anastasia Bernadette
       mean        74.533     69.056
       std dev.     8.999      5.775
    
    Hypotheses: 
       null:        population means are equal, or smaller for group 'Anastasia' 
       alternative: population mean is larger for group 'Anastasia' 
    
    Test results: 
       t-statistic:  2.034 
       degrees of freedom:  23.025 
       p-value:  0.027 
    
    Other information: 
       one-sided 95% confidence interval:  [0.863, Inf] 
       estimated effect size (Cohen's d):  0.724 



check if the grades go up from test 1 to test 2 for Dr. Chico's class


```R
pairedSamplesTTest(formula=~grade_test2+grade_test1, data=chico, one.sided="grade_test2")
```


    
       Paired samples t-test 
    
    Variables:  grade_test2 , grade_test1 
    
    Descriptive statistics: 
                grade_test2 grade_test1 difference
       mean          58.385      56.980      1.405
       std dev.       6.406       6.616      0.970
    
    Hypotheses: 
       null:        population means are equal, or smaller for measurement 'grade_test2' 
       alternative: population mean is larger for measurement 'grade_test2' 
    
    Test results: 
       t-statistic:  6.475 
       degrees of freedom:  19 
       p-value:  <.001 
    
    Other information: 
       one-sided 95% confidence interval:  [1.03, Inf] 
       estimated effect size (Cohen's d):  1.448 



## Using the t.test() function

3 different t-tests so far:<br> 
- One sample
- Independent samples (Student's and Welch's)
- paired samples

R has one function that can run all 4 tests

one sample 


```R
t.test(x=grades, mu=67.5)
```


    
    	One Sample t-test
    
    data:  grades
    t = 2.2547, df = 19, p-value = 0.03615
    alternative hypothesis: true mean is not equal to 67.5
    95 percent confidence interval:
     67.84422 76.75578
    sample estimates:
    mean of x 
         72.3 



independent samples

use var.equal to control whether we're doing a studetn or welch test


```R
t.test(formula=grade~tutor, data=harpo)
```


    
    	Welch Two Sample t-test
    
    data:  grade by tutor
    t = 2.0342, df = 23.025, p-value = 0.05361
    alternative hypothesis: true difference in means is not equal to 0
    95 percent confidence interval:
     -0.09249349 11.04804904
    sample estimates:
     mean in group Anastasia mean in group Bernadette 
                    74.53333                 69.05556 




```R
t.test(formula=grade~tutor, data=harpo, var.equal=TRUE)
```


    
    	Two Sample t-test
    
    data:  grade by tutor
    t = 2.1154, df = 31, p-value = 0.04253
    alternative hypothesis: true difference in means is not equal to 0
    95 percent confidence interval:
      0.1965873 10.7589683
    sample estimates:
     mean in group Anastasia mean in group Bernadette 
                    74.53333                 69.05556 




```R
t.test(x=chico$grade_test2, # variable 1 is the "test2" scores
       y=chico$grade_test1, # variable 2 is the "test1" scores,
       paired=TRUE,         # paired test
)
```


    
    	Paired t-test
    
    data:  chico$grade_test2 and chico$grade_test1
    t = 6.4754, df = 19, p-value = 3.321e-06
    alternative hypothesis: true difference in means is not equal to 0
    95 percent confidence interval:
     0.9508686 1.8591314
    sample estimates:
    mean of the differences 
                      1.405 



## Effect size

the most commonly use measure is **Cohen's d**

primarily definied in the context of an independent samples t-test

calculate something along the lines of: 

![image.png](attachment:image.png)


For clarification purposes, $d$ = anything from a sample, $\delta$ = a theoretical population effect

**Cohen's d from one sample**

![image.png](attachment:image.png)


```R
cohensD(x=grades, mu=67.5)
```


0.504169124037094


>the psychology students in Dr. Zeppo's class are achieving grades (mean = 72.3%) that are about .5 standard devaitions higher than the level you'd expect (67.5%) if they were performing at the same level as other students

**Cohen's d from a Student t test**

several different methods. <br> 
true population effect size:
![image.png](attachment:image.png)

to do the actual calculation, replace everything with their respective estimators
![image-2.png](attachment:image-2.png)

corresponds to `method="pooled"` (default). aka Hedges' g statistic. 

- can also use `method = "x.sd"` or `method = "y.sd"` if you want to only use one of the groups for calculating the sd (called Glass' $\Delta$). Sometimes used when one of the groups is a control group. 

- in usual calculation, divide by $N-2$. If we want to divide by $N$ only (i.e. trying to calculate effect size in sample rather than calculate population effect size). use `method="raw"`

- method based on Hedges and Olkin- small bias in pooled estimation. Use `method="corrected"` to multiply usual value by $(N-3)/(N-2.25)$


```R
cohensD(formula=grade~tutor, data=harpo, method="pooled")
```


0.739561404038266


**Cohen's d from a Welch test**

average the population variances

![image.png](attachment:image.png)
![image-2.png](attachment:image-2.png)
![image-3.png](attachment:image-3.png)

all we need to do is set `method="unequal"`


```R
cohensD(formula=grade~tutor,
        data=harpo,
        method="unequal")
```


0.724499546156234


**Cohen's d from a paired-samples t-test**

to measure effect size relative to distribution of difference scores, use `method="paired"`

![image.png](attachment:image.png)


```R
cohensD(x=chico$grade_test2,
        y=chico$grade_test1,
        method="paired")
```


1.44795152774822


often, need to measure effect size relative to original vars, not difference scores. then use the same method for a welch or student test


```R
cohensD(x=chico$grade_test2, y=chico$grade_test1, method="pooled")
```


0.21576463150673


## Checking the normalty of a sample

any time you think the variable of interest is actually an average of a lot of things, you can usually assume that it will be normally distributed., 

example of a non-normal variable: response time

**QQ plots**

each observation = dot. x=theoretical quantile if it were normally distributed, y=actual quantile. If normal, should form straight line


```R
normal.data <- rnorm(n=100)
hist(x=normal.data)
qqnorm(y=normal.data)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%2013_92_0.png)
    



    
![png](Learning%20Statistics%20with%20R%20chapter%2013_92_1.png)
    


**Shapiro-Wilk tests**

null hypothesis: a set of N observations is normally distributed

![image.png](attachment:image.png)

small W = departure from normality


```R
shapiro.test(x=normal.data)
```


    
    	Shapiro-Wilk normality test
    
    data:  normal.data
    W = 0.99108, p-value = 0.7515



## Testing non-normal data with Wilcoxon test

these tests are **non-parametric** - they don't make any assumptions about the what kind of distribution they're from. This makes them less powerful (higher type II error rate)

**two sample wilcoxon test**

Which group is more awsome? A or B?


```R
load("awesome.Rdata")
print(awesome)
```

       scores group
    1     6.4     A
    2    10.7     A
    3    11.9     A
    4     7.3     A
    5    10.0     A
    6    14.5     B
    7    10.4     B
    8    12.9     B
    9    11.7     B
    10   13.0     B


Construct a table that compares every observation in group A against every observation in group B. whenever A>B, place a check

![image.png](attachment:image.png)

then count number of checkmarks- that's the test statistic. 


```R
wilcox.test(formula=scores~group, data=awesome)
```


    
    	Wilcoxon rank sum test
    
    data:  scores by group
    W = 3, p-value = 0.05556
    alternative hypothesis: true location shift is not equal to 0



`alternative` argument to switch between 1 and 2-sided tests


```R
load("awesome2.Rdata")
score.A
```


<ol class=list-inline>
	<li>6.4</li>
	<li>10.7</li>
	<li>11.9</li>
	<li>7.3</li>
	<li>10</li>
</ol>




```R
score.B
```


<ol class=list-inline>
	<li>14.5</li>
	<li>10.4</li>
	<li>12.9</li>
	<li>11.7</li>
	<li>13</li>
</ol>




```R
wilcox.test(x=score.A, y=score.B)
```


    
    	Wilcoxon rank sum test
    
    data:  score.A and score.B
    W = 3, p-value = 0.05556
    alternative hypothesis: true location shift is not equal to 0



**One sample wilcoxon test**


```R
load("happy.Rdata")
print(happiness)
```

       before after change
    1      30     6    -24
    2      43    29    -14
    3      21    11    -10
    4      24    31      7
    5      23    17     -6
    6      40     2    -38
    7      29    31      2
    8      56    21    -35
    9      38     8    -30
    10     16    21      5


Tabulate positive valued scores against negative valued scores

![image.png](attachment:image.png)


```R
wilcox.test(x=happiness$change, mu=0)
```


    
    	Wilcoxon signed rank test
    
    data:  happiness$change
    V = 7, p-value = 0.03711
    alternative hypothesis: true location is not equal to 0


