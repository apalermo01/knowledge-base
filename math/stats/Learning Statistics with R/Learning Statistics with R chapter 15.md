# Learning Statistics with R - Linear Regression

# What is a linear regression model?

fit data to:
$$
y = mx + b
$$

or: 

$$
\hat Y_i = b_1 X_i + b_0
$$

residuals:
$$
\epsilon_i = Y_i - \hat Y_i
$$

so:

$$
Y_i = b_1 X_i + b_0 + \epsilon_i
$$

# Estimating a linear regression model

ordinary least squares regression - minimize the average of the squares of the residuals

**Using the lm() function**


```R
load("parenthood.Rdata")
```


```R
regression.1 <- lm(formula=dan.grump~dan.sleep,
                   data=parenthood)
print(regression.1)
```

    
    Call:
    lm(formula = dan.grump ~ dan.sleep, data = parenthood)
    
    Coefficients:
    (Intercept)    dan.sleep  
        125.956       -8.937  
    


# Multiple linear regression


$$
Y_i = b_2 X_{i2} + b_1 X_{i1} + b_0 + \epsilon_i
$$


```R
regression.2 <- lm(formula = dan.grump ~ dan.sleep + baby.sleep, data=parenthood)
print(regression.2)
```

    
    Call:
    lm(formula = dan.grump ~ dan.sleep + baby.sleep, data = parenthood)
    
    Coefficients:
    (Intercept)    dan.sleep   baby.sleep  
      125.96557     -8.95025      0.01052  
    


# Quantifying the fit of the regression model

**The $R^2$ value**value

$$
\text{SS}_{res} = \sum_i (Y_i - \hat Y_i)^2
$$

$$
\text{SS}_{tot} = \sum_i (Y_i - \bar Y)^2
$$


```R
X <- parenthood$dan.sleep # predictor
Y <- parenthood$dan.grump # outcome 

# make predictions
Y.pred <- -8.94 * X + 125.97

# squared residuals
SS.resid <- sum( (Y-Y.pred)^2 )
print(SS.resid)

SS.tot <- sum( (Y - mean(Y))^2 )
print(SS.tot)
```

    [1] 1838.722
    [1] 9998.59


convert these residuals into an easily interpretable number

1 = no errors

0 = no different than guessing

$$
R^2 = 1 - \frac{\text{SS}_{res}}{\text{SS}_{tot}}
$$


```R
R.squared <- 1 - (SS.resid/SS.tot)
print(R.squared)
```

    [1] 0.8161018


Porportion of the variance that can be accounted for by the predictor

So our predictor explains 81.6% of the variance

**The relationship between regression and correlation**


```R
r <- cor(X, Y) 
print( r^2 )
```

    [1] 0.8161027


the $R^2$ score is identical to the pearson correlation with one predictor

**The adjusted $R^2$ value**

adding predictors will always cause the $R^2$ value to increase (or stay the same). 

fix:<br> 
for K perdictors and N  observations:

$$
\text{adj.} R^2 = 1 - ( \frac{\text{SS}_{res}}{\text{SS}_{tot}} \times \frac{N-1}{N-K-1})
$$

attempts to account for degrees of freedom

# Hypothesis tests for regression models

- test whether the regression model is performing better than a null model
- test whether a particular regression coefficient is significantly different from 0

**Testing the model as a whole**<br> 

$$
H_0: Y_i = b_0 + \epsilon_i
$$

$$
H_1: Y_i = (\sum_{k=1}^K b_k X_{ik}) + b_0 + \epsilon_i
$$

divide up the total variance into residual variance and regression model variance<br> 

$$
\text{SS}_{mod} = \text{SS}_{tot} - \text{SS}_{res}
$$

convert to mean squares by dividing by degrees of freedom

$$
\text{MS}_{mod} = \frac{\text{SS}_{mod}}{df_{mod}}
$$

$$
\text{MS}_{res} = \frac{\text{SS}_{res}}{df_{res}}
$$

$$
df_{mod} = K
$$

$$
df_{res} = N-K-1
$$

now calculate an F statistic and run an F-test

$$
F = \frac{\text{MS}_{mod}}{\text{MS}_{res}}
$$

**Tests for individual coefficients**


```R
print(regression.2)
```

    
    Call:
    lm(formula = dan.grump ~ dan.sleep + baby.sleep, data = parenthood)
    
    Coefficients:
    (Intercept)    dan.sleep   baby.sleep  
      125.96557     -8.95025      0.01052  
    


Does the baby.sleep coefficient really play any part in determining grump? 

hypotheses:

$$
H_0: b = 0
$$
$$
H_1: b \neq 0
$$

guess that the sampling distribution of $\hat b$ is normal with mean centered on b. if the null were true, then $\hat b$ has a mean 0 and unknown std- so use a t-test!

$$
t = \frac{\hat b}{\text{SE}(\hat b)}
$$

$$
df = N-K-1
$$

estimate of standard error is ugly. 

**Running the hypothesis test in R**


```R
summary(regression.2)
```


    
    Call:
    lm(formula = dan.grump ~ dan.sleep + baby.sleep, data = parenthood)
    
    Residuals:
         Min       1Q   Median       3Q      Max 
    -11.0345  -2.2198  -0.4016   2.6775  11.7496 
    
    Coefficients:
                 Estimate Std. Error t value Pr(>|t|)    
    (Intercept) 125.96557    3.04095  41.423   <2e-16 ***
    dan.sleep    -8.95025    0.55346 -16.172   <2e-16 ***
    baby.sleep    0.01052    0.27106   0.039    0.969    
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 4.354 on 97 degrees of freedom
    Multiple R-squared:  0.8161,	Adjusted R-squared:  0.8123 
    F-statistic: 215.2 on 2 and 97 DF,  p-value: < 2.2e-16



Note: this function used several t-tests without any correction to p-value

## Testing the significance of a correlation

**Hypothesis test for a single correlation**


```R
summary(regression.1)
```


    
    Call:
    lm(formula = dan.grump ~ dan.sleep, data = parenthood)
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -11.025  -2.213  -0.399   2.681  11.750 
    
    Coefficients:
                Estimate Std. Error t value Pr(>|t|)    
    (Intercept) 125.9563     3.0161   41.76   <2e-16 ***
    dan.sleep    -8.9368     0.4285  -20.85   <2e-16 ***
    ---
    Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
    
    Residual standard error: 4.332 on 98 degrees of freedom
    Multiple R-squared:  0.8161,	Adjusted R-squared:  0.8142 
    F-statistic: 434.9 on 1 and 98 DF,  p-value: < 2.2e-16




```R
cor.test(x=parenthood$dan.sleep, y=parenthood$dan.grump)
```


    
    	Pearson's product-moment correlation
    
    data:  parenthood$dan.sleep and parenthood$dan.grump
    t = -20.854, df = 98, p-value < 2.2e-16
    alternative hypothesis: true correlation is not equal to 0
    95 percent confidence interval:
     -0.9340614 -0.8594714
    sample estimates:
          cor 
    -0.903384 



these two are exactly the same test

**Hypothesis test for all pairwise correlations**

cor.test() won't work on multiple pairs of variables becuase running many hypothesis tests is dangerous

correlate() in lsr package will apply corrections to p values


```R
library(lsr)
correlate(parenthood, test=TRUE)
```


<dl>
	<dt>$correlation</dt>
		<dd><table>
<thead><tr><th></th><th scope=col>dan.sleep</th><th scope=col>baby.sleep</th><th scope=col>dan.grump</th><th scope=col>day</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td>         NA</td><td> 0.62794934</td><td>-0.90338404</td><td>-0.09840768</td></tr>
	<tr><th scope=row>baby.sleep</th><td> 0.62794934</td><td>         NA</td><td>-0.56596373</td><td>-0.01043394</td></tr>
	<tr><th scope=row>dan.grump</th><td>-0.90338404</td><td>-0.56596373</td><td>         NA</td><td> 0.07647926</td></tr>
	<tr><th scope=row>day</th><td>-0.09840768</td><td>-0.01043394</td><td> 0.07647926</td><td>         NA</td></tr>
</tbody>
</table>
</dd>
	<dt>$p.value</dt>
		<dd><table>
<thead><tr><th></th><th scope=col>dan.sleep</th><th scope=col>baby.sleep</th><th scope=col>dan.grump</th><th scope=col>day</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td>          NA</td><td>1.348133e-11</td><td>4.905856e-37</td><td>0.9900633   </td></tr>
	<tr><th scope=row>baby.sleep</th><td>1.348133e-11</td><td>          NA</td><td>3.379004e-09</td><td>0.9900633   </td></tr>
	<tr><th scope=row>dan.grump</th><td>4.905856e-37</td><td>3.379004e-09</td><td>          NA</td><td>0.9900633   </td></tr>
	<tr><th scope=row>day</th><td>9.900633e-01</td><td>9.900633e-01</td><td>9.900633e-01</td><td>       NA   </td></tr>
</tbody>
</table>
</dd>
	<dt>$sample.size</dt>
		<dd><table>
<thead><tr><th></th><th scope=col>dan.sleep</th><th scope=col>baby.sleep</th><th scope=col>dan.grump</th><th scope=col>day</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td>100</td><td>100</td><td>100</td><td>100</td></tr>
	<tr><th scope=row>baby.sleep</th><td>100</td><td>100</td><td>100</td><td>100</td></tr>
	<tr><th scope=row>dan.grump</th><td>100</td><td>100</td><td>100</td><td>100</td></tr>
	<tr><th scope=row>day</th><td>100</td><td>100</td><td>100</td><td>100</td></tr>
</tbody>
</table>
</dd>
	<dt>$args</dt>
		<dd><dl class=dl-horizontal>
	<dt>two.inputs</dt>
		<dd>'FALSE'</dd>
	<dt>test</dt>
		<dd>'TRUE'</dd>
	<dt>corr.method</dt>
		<dd>'pearson'</dd>
	<dt>p.adjust.method</dt>
		<dd>'holm'</dd>
</dl>
</dd>
	<dt>$tiesProblem</dt>
		<dd>FALSE</dd>
</dl>



# Regarding regression coefficients

**Confidence intervals for the coefficients**

construct them in the usual way:

$$
\text{CI}(b) = \hat b \pm (t_{crit} \times \text{SE}(\hat b))
$$

use `confint()` function


```R
confint(
    object=regression.2, # regression model
    # parm - vector indicating which coefficients to get the interval for, default = all
    level=0.99,
)
```


<table>
<thead><tr><th></th><th scope=col>0.5 %</th><th scope=col>99.5 %</th></tr></thead>
<tbody>
	<tr><th scope=row>(Intercept)</th><td>117.9755724</td><td>133.9555593</td></tr>
	<tr><th scope=row>dan.sleep</th><td>-10.4044419</td><td> -7.4960575</td></tr>
	<tr><th scope=row>baby.sleep</th><td> -0.7016868</td><td>  0.7227357</td></tr>
</tbody>
</table>



**Calculating standardized regression coefficients**

get all dependent variables on the same scale. These are the coefficients we would have obtained if we converted all the variables to z-scores before doing the regression. 

$$
\beta_X = b_x \times \frac{\sigma_X}{\sigma_Y}
$$

can use `standardCoefs()` function in lsr package


```R
standardCoefs(regression.2)
```


<table>
<thead><tr><th></th><th scope=col>b</th><th scope=col>beta</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td>-8.95024973</td><td>-0.90474809</td></tr>
	<tr><th scope=row>baby.sleep</th><td> 0.01052447</td><td> 0.00217223</td></tr>
</tbody>
</table>



## Assumptions of regression

- residuals are normally distributed
- the relationship between X and Y is actually linear
- each residual is generated from the same normal distribution with mean 0 and some variance. We want the standard deviation of the residual to be the same for all values of $\hat Y$ and all values of every predictor $X$
- Uncorrelated predictors- we don't want the predictors to be too strongly correlated in a multiple regression model (collinearity)
- Residuals are independent of each other
- No "bad" outliers

## Model checking

This is known as **Regression diagnostics**<br> 

**3 kinds of residuals**

- ordinary residuals: $\epsilon_i = Y_i - \hat Y_i$
- standardised residuals: normalize to have std of 1: $\epsilon_i` = \frac{\epsilon_i}{\hat \sigma \sqrt{1-h_i}}$($\hat \sigma$ = estimated pop std fo ordinary residuals, $h_i$ = 'hat value' of observation (not covered yet)
- studentised residuals: $\epsilon_i` = \frac{\epsilon_i}{\hat \sigma_{(-i)} \sqrt{1-h_i}}$, $\hat \sigma_{(-i)}$ = estimate of std if you had eleminated that observation from the dataset

![image.png](attachment:image.png)

- pearson residual (identical to ordinary residual for our purposes)


```R
# ordinary residuals
residuals(object=regression.2)
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>-2.14030951078877</dd>
	<dt>2</dt>
		<dd>4.70819418602325</dd>
	<dt>3</dt>
		<dd>1.95536395121787</dd>
	<dt>4</dt>
		<dd>-2.06028059439638</dd>
	<dt>5</dt>
		<dd>0.719488755895294</dd>
	<dt>6</dt>
		<dd>-0.406613299737074</dd>
	<dt>7</dt>
		<dd>0.226998721062952</dd>
	<dt>8</dt>
		<dd>-1.70030766353261</dd>
	<dt>9</dt>
		<dd>0.202503861471201</dd>
	<dt>10</dt>
		<dd>3.85245887607994</dd>
	<dt>11</dt>
		<dd>3.99862907746025</dd>
	<dt>12</dt>
		<dd>-4.9120150484327</dd>
	<dt>13</dt>
		<dd>1.20601343637333</dd>
	<dt>14</dt>
		<dd>0.494657819377132</dd>
	<dt>15</dt>
		<dd>-2.65792763886342</dd>
	<dt>16</dt>
		<dd>-0.396680475175301</dd>
	<dt>17</dt>
		<dd>3.35386132572363</dd>
	<dt>18</dt>
		<dd>1.72612249497873</dd>
	<dt>19</dt>
		<dd>-0.492255128414457</dd>
	<dt>20</dt>
		<dd>-5.64059409024558</dd>
	<dt>21</dt>
		<dd>-0.466076353432413</dd>
	<dt>22</dt>
		<dd>2.72383894016154</dd>
	<dt>23</dt>
		<dd>9.36536970725366</dd>
	<dt>24</dt>
		<dd>0.284151270185918</dd>
	<dt>25</dt>
		<dd>-0.503766832178665</dd>
	<dt>26</dt>
		<dd>-1.49411459067392</dd>
	<dt>27</dt>
		<dd>8.13286234271166</dd>
	<dt>28</dt>
		<dd>1.97873159949224</dd>
	<dt>29</dt>
		<dd>-1.51267260252164</dd>
	<dt>30</dt>
		<dd>3.51711476465414</dd>
	<dt>31</dt>
		<dd>-8.92569509559963</dd>
	<dt>32</dt>
		<dd>-2.82829457432458</dd>
	<dt>33</dt>
		<dd>6.10303494536796</dd>
	<dt>34</dt>
		<dd>-7.54607167560131</dd>
	<dt>35</dt>
		<dd>4.55721279136122</dd>
	<dt>36</dt>
		<dd>-10.6510836188802</dd>
	<dt>37</dt>
		<dd>-5.69318461734748</dd>
	<dt>38</dt>
		<dd>6.30965055392929</dd>
	<dt>39</dt>
		<dd>-2.10824658159528</dd>
	<dt>40</dt>
		<dd>-0.504425250205221</dd>
	<dt>41</dt>
		<dd>0.187557555527359</dd>
	<dt>42</dt>
		<dd>4.80948409107311</dd>
	<dt>43</dt>
		<dd>-5.41351630684395</dd>
	<dt>44</dt>
		<dd>-6.2292842276922</dd>
	<dt>45</dt>
		<dd>-4.57252324936502</dd>
	<dt>46</dt>
		<dd>-5.33546009022224</dd>
	<dt>47</dt>
		<dd>3.99501114413609</dd>
	<dt>48</dt>
		<dd>2.17187453660028</dd>
	<dt>49</dt>
		<dd>-3.47664396849973</dd>
	<dt>50</dt>
		<dd>0.483436665372514</dd>
	<dt>51</dt>
		<dd>6.28397904555092</dd>
	<dt>52</dt>
		<dd>2.01093959504155</dd>
	<dt>53</dt>
		<dd>-1.58466309916641</dd>
	<dt>54</dt>
		<dd>-2.2166612982094</dd>
	<dt>55</dt>
		<dd>2.20331399032188</dd>
	<dt>56</dt>
		<dd>1.93287362024133</dd>
	<dt>57</dt>
		<dd>-1.83012044236047</dd>
	<dt>58</dt>
		<dd>-1.54014302918071</dd>
	<dt>59</dt>
		<dd>2.52985093164823</dd>
	<dt>60</dt>
		<dd>-3.37057822977084</dd>
	<dt>61</dt>
		<dd>-2.93808058590344</dd>
	<dt>62</dt>
		<dd>0.659073620703663</dd>
	<dt>63</dt>
		<dd>-0.591755873362145</dd>
	<dt>64</dt>
		<dd>-8.61319707945395</dd>
	<dt>65</dt>
		<dd>5.97810345345508</dd>
	<dt>66</dt>
		<dd>5.9332979213231</dd>
	<dt>67</dt>
		<dd>-1.2341955780725</dd>
	<dt>68</dt>
		<dd>3.0047668649736</dd>
	<dt>69</dt>
		<dd>-1.08024681766661</dd>
	<dt>70</dt>
		<dd>6.51746720672353</dd>
	<dt>71</dt>
		<dd>-3.01554692127676</dd>
	<dt>72</dt>
		<dd>2.11767195312046</dd>
	<dt>73</dt>
		<dd>0.605875669899511</dd>
	<dt>74</dt>
		<dd>-2.72374208320604</dd>
	<dt>75</dt>
		<dd>-2.22914715413585</dd>
	<dt>76</dt>
		<dd>-1.40538219203882</dd>
	<dt>77</dt>
		<dd>4.74614905370369</dd>
	<dt>78</dt>
		<dd>11.7495569146522</dd>
	<dt>79</dt>
		<dd>4.76341406554335</dd>
	<dt>80</dt>
		<dd>2.66209077479395</dd>
	<dt>81</dt>
		<dd>-11.0345292342062</dd>
	<dt>82</dt>
		<dd>-0.758866658698959</dd>
	<dt>83</dt>
		<dd>1.45582272921276</dd>
	<dt>84</dt>
		<dd>-0.474572668485457</dd>
	<dt>85</dt>
		<dd>8.90912014419803</dd>
	<dt>86</dt>
		<dd>-1.14097768699392</dd>
	<dt>87</dt>
		<dd>0.755522269006462</dd>
	<dt>88</dt>
		<dd>-0.410712964411633</dd>
	<dt>89</dt>
		<dd>0.879723692404491</dd>
	<dt>90</dt>
		<dd>-1.409558594815</dd>
	<dt>91</dt>
		<dd>3.15713851155618</dd>
	<dt>92</dt>
		<dd>-3.42057569534016</dd>
	<dt>93</dt>
		<dd>-5.72286985352152</dd>
	<dt>94</dt>
		<dd>-2.2033957939457</dd>
	<dt>95</dt>
		<dd>-3.86478909645122</dd>
	<dt>96</dt>
		<dd>0.498271082136077</dd>
	<dt>97</dt>
		<dd>-5.52494952587938</dd>
	<dt>98</dt>
		<dd>4.11342213447684</dd>
	<dt>99</dt>
		<dd>-8.20385327220295</dd>
	<dt>100</dt>
		<dd>5.68008585514849</dd>
</dl>




```R
# standardised residuals
rstandard(model=regression.2)
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>-0.496758449499741</dd>
	<dt>2</dt>
		<dd>1.10430570922009</dd>
	<dt>3</dt>
		<dd>0.463612644165956</dd>
	<dt>4</dt>
		<dd>-0.477253571698728</dd>
	<dt>5</dt>
		<dd>0.16756280555839</dd>
	<dt>6</dt>
		<dd>-0.0948896859665602</dd>
	<dt>7</dt>
		<dd>0.0528662634165132</dd>
	<dt>8</dt>
		<dd>-0.392603807260774</dd>
	<dt>9</dt>
		<dd>0.0473969139556533</dd>
	<dt>10</dt>
		<dd>0.890339897416215</dd>
	<dt>11</dt>
		<dd>0.958512482604786</dd>
	<dt>12</dt>
		<dd>-1.13898701499646</dd>
	<dt>13</dt>
		<dd>0.280478412328987</dd>
	<dt>14</dt>
		<dd>0.115191838021472</dd>
	<dt>15</dt>
		<dd>-0.616570915779299</dd>
	<dt>16</dt>
		<dd>-0.0919186506458024</dd>
	<dt>17</dt>
		<dd>0.776929365788282</dd>
	<dt>18</dt>
		<dd>0.404034948280612</dd>
	<dt>19</dt>
		<dd>-0.1155237268069</dd>
	<dt>20</dt>
		<dd>-1.31540411792587</dd>
	<dt>21</dt>
		<dd>-0.108192380975879</dd>
	<dt>22</dt>
		<dd>0.629518238164769</dd>
	<dt>23</dt>
		<dd>2.17129803304802</dd>
	<dt>24</dt>
		<dd>0.065862268161991</dd>
	<dt>25</dt>
		<dd>-0.119804485507872</dd>
	<dt>26</dt>
		<dd>-0.347040238403693</dd>
	<dt>27</dt>
		<dd>1.91121832986674</dd>
	<dt>28</dt>
		<dd>0.456865160555886</dd>
	<dt>29</dt>
		<dd>-0.34986350282249</dd>
	<dt>30</dt>
		<dd>0.812331649663395</dd>
	<dt>31</dt>
		<dd>-2.08659992694796</dd>
	<dt>32</dt>
		<dd>-0.663178432178709</dd>
	<dt>33</dt>
		<dd>1.42930082458654</dd>
	<dt>34</dt>
		<dd>-1.77763064379685</dd>
	<dt>35</dt>
		<dd>1.07452435879932</dd>
	<dt>36</dt>
		<dd>-2.47385779544741</dd>
	<dt>37</dt>
		<dd>-1.32715113878353</dd>
	<dt>38</dt>
		<dd>1.49419657696974</dd>
	<dt>39</dt>
		<dd>-0.491156390547011</dd>
	<dt>40</dt>
		<dd>-0.116749474464975</dd>
	<dt>41</dt>
		<dd>0.0440123325402083</dd>
	<dt>42</dt>
		<dd>1.11881912165588</dd>
	<dt>43</dt>
		<dd>-1.270816414813</dd>
	<dt>44</dt>
		<dd>-1.46422594954453</dd>
	<dt>45</dt>
		<dd>-1.06943700351355</dd>
	<dt>46</dt>
		<dd>-1.24659673042949</dd>
	<dt>47</dt>
		<dd>0.941528809720747</dd>
	<dt>48</dt>
		<dd>0.510698087753031</dd>
	<dt>49</dt>
		<dd>-0.813733492026347</dd>
	<dt>50</dt>
		<dd>0.114121782027834</dd>
	<dt>51</dt>
		<dd>1.47938594037061</dd>
	<dt>52</dt>
		<dd>0.464379621041514</dd>
	<dt>53</dt>
		<dd>-0.371570089154635</dd>
	<dt>54</dt>
		<dd>-0.516099490850526</dd>
	<dt>55</dt>
		<dd>0.518007531168239</dd>
	<dt>56</dt>
		<dd>0.448132039610816</dd>
	<dt>57</dt>
		<dd>-0.426623584277989</dd>
	<dt>58</dt>
		<dd>-0.355756105445386</dd>
	<dt>59</dt>
		<dd>0.584032967679746</dd>
	<dt>60</dt>
		<dd>-0.780226766217168</dd>
	<dt>61</dt>
		<dd>-0.678333246916289</dd>
	<dt>62</dt>
		<dd>0.154846993568762</dd>
	<dt>63</dt>
		<dd>-0.137605736546182</dd>
	<dt>64</dt>
		<dd>-2.05662231903212</dd>
	<dt>65</dt>
		<dd>1.40238029319466</dd>
	<dt>66</dt>
		<dd>1.37505124853628</dd>
	<dt>67</dt>
		<dd>-0.28964989305473</dd>
	<dt>68</dt>
		<dd>0.694976317173236</dd>
	<dt>69</dt>
		<dd>-0.249453163358788</dd>
	<dt>70</dt>
		<dd>1.50709622764506</dd>
	<dt>71</dt>
		<dd>-0.698646815336446</dd>
	<dt>72</dt>
		<dd>0.490714272270967</dd>
	<dt>73</dt>
		<dd>0.142672972955054</dd>
	<dt>74</dt>
		<dd>-0.632465601965175</dd>
	<dt>75</dt>
		<dd>-0.519728279993225</dd>
	<dt>76</dt>
		<dd>-0.325098106205228</dd>
	<dt>77</dt>
		<dd>1.10842573618794</dd>
	<dt>78</dt>
		<dd>2.72171670657618</dd>
	<dt>79</dt>
		<dd>1.0997510145679</dd>
	<dt>80</dt>
		<dd>0.620570796899556</dd>
	<dt>81</dt>
		<dd>-2.55172096627599</dd>
	<dt>82</dt>
		<dd>-0.175848025719297</dd>
	<dt>83</dt>
		<dd>0.343400636794751</dd>
	<dt>84</dt>
		<dd>-0.111589524340065</dd>
	<dt>85</dt>
		<dd>2.10863391493908</dd>
	<dt>86</dt>
		<dd>-0.263865155911573</dd>
	<dt>87</dt>
		<dd>0.176244453297667</dd>
	<dt>88</dt>
		<dd>-0.0950441612721027</dd>
	<dt>89</dt>
		<dd>0.204508843594006</dd>
	<dt>90</dt>
		<dd>-0.32730740367351</dd>
	<dt>91</dt>
		<dd>0.734756402816328</dd>
	<dt>92</dt>
		<dd>-0.794008551500343</dd>
	<dt>93</dt>
		<dd>-1.32768247621787</dd>
	<dt>94</dt>
		<dd>-0.519407357692693</dd>
	<dt>95</dt>
		<dd>-0.91512580224067</dd>
	<dt>96</dt>
		<dd>0.116612262727703</dd>
	<dt>97</dt>
		<dd>-1.28069114688173</dd>
	<dt>98</dt>
		<dd>0.96332848822972</dd>
	<dt>99</dt>
		<dd>-1.90290257908238</dd>
	<dt>100</dt>
		<dd>1.31368144195261</dd>
</dl>




```R
# studentised residuals
rstudent(model=regression.2)
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>-0.494821020783785</dd>
	<dt>2</dt>
		<dd>1.105570301427</dd>
	<dt>3</dt>
		<dd>0.461728539615688</dd>
	<dt>4</dt>
		<dd>-0.47534554882598</dd>
	<dt>5</dt>
		<dd>0.166720973102369</dd>
	<dt>6</dt>
		<dd>-0.0944036783355519</dd>
	<dt>7</dt>
		<dd>0.0525938086168564</dd>
	<dt>8</dt>
		<dd>-0.390885525028329</dd>
	<dt>9</dt>
		<dd>0.047152513040637</dd>
	<dt>10</dt>
		<dd>0.889380186324829</dd>
	<dt>11</dt>
		<dd>0.958107100330896</dd>
	<dt>12</dt>
		<dd>-1.14075472102211</dd>
	<dt>13</dt>
		<dd>0.279142118515376</dd>
	<dt>14</dt>
		<dd>0.114604366326401</dd>
	<dt>15</dt>
		<dd>-0.614590005774153</dd>
	<dt>16</dt>
		<dd>-0.0914475984686252</dd>
	<dt>17</dt>
		<dd>0.775330358206955</dd>
	<dt>18</dt>
		<dd>0.402285550207554</dd>
	<dt>19</dt>
		<dd>-0.114934607869236</dd>
	<dt>20</dt>
		<dd>-1.32043609464376</dd>
	<dt>21</dt>
		<dd>-0.107639738454238</dd>
	<dt>22</dt>
		<dd>0.62754812716805</dd>
	<dt>23</dt>
		<dd>2.21456485489445</dd>
	<dt>24</dt>
		<dd>0.065523357522646</dd>
	<dt>25</dt>
		<dd>-0.11919415565622</dd>
	<dt>26</dt>
		<dd>-0.34546126883684</dd>
	<dt>27</dt>
		<dd>1.93818472778376</dd>
	<dt>28</dt>
		<dd>0.454993878786888</dd>
	<dt>29</dt>
		<dd>-0.348275224527629</dd>
	<dt>30</dt>
		<dd>0.810896461863637</dd>
	<dt>31</dt>
		<dd>-2.1240328615166</dd>
	<dt>32</dt>
		<dd>-0.661251917734998</dd>
	<dt>33</dt>
		<dd>1.43712829909911</dd>
	<dt>34</dt>
		<dd>-1.79797263472033</dd>
	<dt>35</dt>
		<dd>1.07539063576337</dd>
	<dt>36</dt>
		<dd>-2.54258876054003</dd>
	<dt>37</dt>
		<dd>-1.3324451487556</dd>
	<dt>38</dt>
		<dd>1.5038825651503</dd>
	<dt>39</dt>
		<dd>-0.489226818789108</dd>
	<dt>40</dt>
		<dd>-0.116154275254689</dd>
	<dt>41</dt>
		<dd>0.0437853143001301</dd>
	<dt>42</dt>
		<dd>1.12028904298392</dd>
	<dt>43</dt>
		<dd>-1.27490649017439</dd>
	<dt>44</dt>
		<dd>-1.47302872318343</dd>
	<dt>45</dt>
		<dd>-1.0702382848786</dd>
	<dt>46</dt>
		<dd>-1.25020934703487</dd>
	<dt>47</dt>
		<dd>0.940972606895438</dd>
	<dt>48</dt>
		<dd>0.508743215441837</dd>
	<dt>49</dt>
		<dd>-0.812305437486815</dd>
	<dt>50</dt>
		<dd>0.113539623927593</dd>
	<dt>51</dt>
		<dd>1.48863005930873</dd>
	<dt>52</dt>
		<dd>0.46249410009877</dd>
	<dt>53</dt>
		<dd>-0.369913167957844</dd>
	<dt>54</dt>
		<dd>-0.514138681049772</dd>
	<dt>55</dt>
		<dd>0.516044735196355</dd>
	<dt>56</dt>
		<dd>0.446278308020155</dd>
	<dt>57</dt>
		<dd>-0.424817540598925</dd>
	<dt>58</dt>
		<dd>-0.354148676472633</dd>
	<dt>59</dt>
		<dd>0.582038942458752</dd>
	<dt>60</dt>
		<dd>-0.778641709725079</dd>
	<dt>61</dt>
		<dd>-0.676433922138623</dd>
	<dt>62</dt>
		<dd>0.15406578841705</dd>
	<dt>63</dt>
		<dd>-0.136907954559264</dd>
	<dt>64</dt>
		<dd>-2.0921155609947</dd>
	<dt>65</dt>
		<dd>1.40949469036901</dd>
	<dt>66</dt>
		<dd>1.38147541060835</dd>
	<dt>67</dt>
		<dd>-0.288277679629173</dd>
	<dt>68</dt>
		<dd>0.693112445031682</dd>
	<dt>69</dt>
		<dd>-0.248243629779167</dd>
	<dt>70</dt>
		<dd>1.51717577666183</dd>
	<dt>71</dt>
		<dd>-0.696791564028876</dd>
	<dt>72</dt>
		<dd>0.48878534097895</dd>
	<dt>73</dt>
		<dd>0.14195053502797</dd>
	<dt>74</dt>
		<dd>-0.630498405169727</dd>
	<dt>75</dt>
		<dd>-0.517763742911117</dd>
	<dt>76</dt>
		<dd>-0.323594339589895</dd>
	<dt>77</dt>
		<dd>1.10974786130888</dd>
	<dt>78</dt>
		<dd>2.81736616123487</dd>
	<dt>79</dt>
		<dd>1.10095269677747</dd>
	<dt>80</dt>
		<dd>0.618592877065821</dd>
	<dt>81</dt>
		<dd>-2.62827967020593</dd>
	<dt>82</dt>
		<dd>-0.174967135326626</dd>
	<dt>83</dt>
		<dd>0.341833793416068</dd>
	<dt>84</dt>
		<dd>-0.111019956665829</dd>
	<dt>85</dt>
		<dd>2.14753375145234</dd>
	<dt>86</dt>
		<dd>-0.262595762694396</dd>
	<dt>87</dt>
		<dd>0.17536170325635</dd>
	<dt>88</dt>
		<dd>-0.0945573767487656</dd>
	<dt>89</dt>
		<dd>0.203495819042005</dd>
	<dt>90</dt>
		<dd>-0.325795840988614</dd>
	<dt>91</dt>
		<dd>0.733001844607368</dd>
	<dt>92</dt>
		<dd>-0.792484688230581</dd>
	<dt>93</dt>
		<dd>-1.33298847740706</dd>
	<dt>94</dt>
		<dd>-0.517443141718129</dd>
	<dt>95</dt>
		<dd>-0.914352050771026</dd>
	<dt>96</dt>
		<dd>0.116017743883484</dd>
	<dt>97</dt>
		<dd>-1.28498272562851</dd>
	<dt>98</dt>
		<dd>0.96296745202377</dd>
	<dt>99</dt>
		<dd>-1.92942388621584</dd>
	<dt>100</dt>
		<dd>1.31867548457541</dd>
</dl>



**Three kinds of anomalous data**<br> 

- outlier: something that is very different from what your model predicts (very large studentised residual). Could mean junk data or some other defect OR something to look more into
- leverage: an observation is very different from all other observations. use hat value- measure to the extend to whihc the i-th observation is in control of where the regression line ends up going. Worth looking into, but less of a cause for concern unless they're also outliers.

![image.png](attachment:image.png)

- influence- outlier that has a high leverage

![image-2.png](attachment:image-2.png)

measure using cook's distance 

![image-3.png](attachment:image-3.png)

in general, large if > 1 or > 4/N


```R
# get hat values
hatvalues(model=regression.2)
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>0.0206745229958039</dd>
	<dt>2</dt>
		<dd>0.0410531998824756</dd>
	<dt>3</dt>
		<dd>0.061554447448945</dd>
	<dt>4</dt>
		<dd>0.0168522600195781</dd>
	<dt>5</dt>
		<dd>0.0273486549917566</dd>
	<dt>6</dt>
		<dd>0.03129942733896</dd>
	<dt>7</dt>
		<dd>0.0273557911853968</dd>
	<dt>8</dt>
		<dd>0.0105122392809176</dd>
	<dt>9</dt>
		<dd>0.0369897645326393</dd>
	<dt>10</dt>
		<dd>0.012291551997734</dd>
	<dt>11</dt>
		<dd>0.0818976281364723</dd>
	<dt>12</dt>
		<dd>0.0188255123468677</dd>
	<dt>13</dt>
		<dd>0.0246290249560646</dd>
	<dt>14</dt>
		<dd>0.0271838830380095</dd>
	<dt>15</dt>
		<dd>0.0196421002741052</dd>
	<dt>16</dt>
		<dd>0.017485915584454</dd>
	<dt>17</dt>
		<dd>0.0169139234913173</dd>
	<dt>18</dt>
		<dd>0.0371252996435166</dd>
	<dt>19</dt>
		<dd>0.0421389074651149</dd>
	<dt>20</dt>
		<dd>0.0299464312958827</dd>
	<dt>21</dt>
		<dd>0.0209943507017963</dd>
	<dt>22</dt>
		<dd>0.012332801267656</dd>
	<dt>23</dt>
		<dd>0.018533703706706</dd>
	<dt>24</dt>
		<dd>0.0180480123017897</dd>
	<dt>25</dt>
		<dd>0.0672239221297222</dd>
	<dt>26</dt>
		<dd>0.0221492718654132</dd>
	<dt>27</dt>
		<dd>0.0447200669644985</dd>
	<dt>28</dt>
		<dd>0.010394472621875</dd>
	<dt>29</dt>
		<dd>0.0138181214043978</dd>
	<dt>30</dt>
		<dd>0.0110581707673313</dd>
	<dt>31</dt>
		<dd>0.0346826001902454</dd>
	<dt>32</dt>
		<dd>0.0404824794939356</dd>
	<dt>33</dt>
		<dd>0.0381466970169939</dd>
	<dt>34</dt>
		<dd>0.0493443990717182</dd>
	<dt>35</dt>
		<dd>0.0510780317672356</dd>
	<dt>36</dt>
		<dd>0.0220817719677087</dd>
	<dt>37</dt>
		<dd>0.0291901318947565</dd>
	<dt>38</dt>
		<dd>0.0592817781553077</dd>
	<dt>39</dt>
		<dd>0.0279969510493665</dd>
	<dt>40</dt>
		<dd>0.0151996747081158</dd>
	<dt>41</dt>
		<dd>0.0419575069254178</dd>
	<dt>42</dt>
		<dd>0.0251413707700619</dd>
	<dt>43</dt>
		<dd>0.0426787863052519</dd>
	<dt>44</dt>
		<dd>0.0451733950195895</dd>
	<dt>45</dt>
		<dd>0.0355808030376597</dd>
	<dt>46</dt>
		<dd>0.0336016045101932</dd>
	<dt>47</dt>
		<dd>0.0501977826710969</dd>
	<dt>48</dt>
		<dd>0.0458746830659391</dd>
	<dt>49</dt>
		<dd>0.0370128988966657</dd>
	<dt>50</dt>
		<dd>0.0533128159891432</dd>
	<dt>51</dt>
		<dd>0.0481447749870646</dd>
	<dt>52</dt>
		<dd>0.0107269889235381</dd>
	<dt>53</dt>
		<dd>0.0404738558583937</dd>
	<dt>54</dt>
		<dd>0.0268131496328796</dd>
	<dt>55</dt>
		<dd>0.0455678672533901</dd>
	<dt>56</dt>
		<dd>0.0185699688891927</dd>
	<dt>57</dt>
		<dd>0.0291904502312289</dd>
	<dt>58</dt>
		<dd>0.011260690937101</dd>
	<dt>59</dt>
		<dd>0.0101268298410569</dd>
	<dt>60</dt>
		<dd>0.0154641186320434</dd>
	<dt>61</dt>
		<dd>0.0102953385761676</dd>
	<dt>62</dt>
		<dd>0.0442887038636444</dd>
	<dt>63</dt>
		<dd>0.0243894420435451</dd>
	<dt>64</dt>
		<dd>0.0746967328787712</dd>
	<dt>65</dt>
		<dd>0.0413509016314271</dd>
	<dt>66</dt>
		<dd>0.0177569660370351</dd>
	<dt>67</dt>
		<dd>0.0421761584990673</dd>
	<dt>68</dt>
		<dd>0.0138432128441322</dd>
	<dt>69</dt>
		<dd>0.0106900531366874</dd>
	<dt>70</dt>
		<dd>0.0134021616508365</dd>
	<dt>71</dt>
		<dd>0.0171636075960713</dd>
	<dt>72</dt>
		<dd>0.0175184434857562</dd>
	<dt>73</dt>
		<dd>0.0486331386176523</dd>
	<dt>74</dt>
		<dd>0.0215862286302109</dd>
	<dt>75</dt>
		<dd>0.0295141761516455</dd>
	<dt>76</dt>
		<dd>0.0141191542672195</dd>
	<dt>77</dt>
		<dd>0.0327606416614395</dd>
	<dt>78</dt>
		<dd>0.0168459870127456</dd>
	<dt>79</dt>
		<dd>0.0102800086376359</dd>
	<dt>80</dt>
		<dd>0.0292051412685951</dd>
	<dt>81</dt>
		<dd>0.0134805071901966</dd>
	<dt>82</dt>
		<dd>0.0175275821101051</dd>
	<dt>83</dt>
		<dd>0.0518452728318131</dd>
	<dt>84</dt>
		<dd>0.0458360400378517</dd>
	<dt>85</dt>
		<dd>0.0582585793475386</dd>
	<dt>86</dt>
		<dd>0.0135964419880705</dd>
	<dt>87</dt>
		<dd>0.0305441417220504</dd>
	<dt>88</dt>
		<dd>0.014877236816051</dd>
	<dt>89</dt>
		<dd>0.0238134791400542</dd>
	<dt>90</dt>
		<dd>0.0215941840052774</dd>
	<dt>91</dt>
		<dd>0.025986606050814</dd>
	<dt>92</dt>
		<dd>0.0209328768841424</dd>
	<dt>93</dt>
		<dd>0.0198248030918108</dd>
	<dt>94</dt>
		<dd>0.0506349199822207</dd>
	<dt>95</dt>
		<dd>0.0590762882487533</dd>
	<dt>96</dt>
		<dd>0.036820264076811</dd>
	<dt>97</dt>
		<dd>0.0181791948616895</dd>
	<dt>98</dt>
		<dd>0.0381171760726119</dd>
	<dt>99</dt>
		<dd>0.0194560261521814</dd>
	<dt>100</dt>
		<dd>0.0137339435701813</dd>
</dl>




```R
# calculate cooks distances
cooks.distance(model=regression.2)
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>0.00173651171104773</dd>
	<dt>2</dt>
		<dd>0.0174024293642054</dd>
	<dt>3</dt>
		<dd>0.00469937006422766</dd>
	<dt>4</dt>
		<dd>0.00130141703153831</dd>
	<dt>5</dt>
		<dd>0.000263155694778566</dd>
	<dt>6</dt>
		<dd>9.69758509568525e-05</dd>
	<dt>7</dt>
		<dd>2.62018074689963e-05</dd>
	<dt>8</dt>
		<dd>0.000545849064210991</dd>
	<dt>9</dt>
		<dd>2.87626925926391e-05</dd>
	<dt>10</dt>
		<dd>0.0032882768121844</dd>
	<dt>11</dt>
		<dd>0.0273183525972684</dd>
	<dt>12</dt>
		<dd>0.00829691896743518</dd>
	<dt>13</dt>
		<dd>0.000662147916887141</dd>
	<dt>14</dt>
		<dd>0.000123595567837425</dd>
	<dt>15</dt>
		<dd>0.00253891455880253</dd>
	<dt>16</dt>
		<dd>5.01228340187192e-05</dd>
	<dt>17</dt>
		<dd>0.00346174150886355</dd>
	<dt>18</dt>
		<dd>0.00209805467653322</dd>
	<dt>19</dt>
		<dd>0.000195704974772914</dd>
	<dt>20</dt>
		<dd>0.0178051852632398</dd>
	<dt>21</dt>
		<dd>8.36737728483593e-05</dd>
	<dt>22</dt>
		<dd>0.00164947782540362</dd>
	<dt>23</dt>
		<dd>0.0296759375517295</dd>
	<dt>24</dt>
		<dd>2.65760991710286e-05</dd>
	<dt>25</dt>
		<dd>0.000344803249503167</dd>
	<dt>26</dt>
		<dd>0.000909337919577246</dd>
	<dt>27</dt>
		<dd>0.0569995122593629</dd>
	<dt>28</dt>
		<dd>0.000730794339415602</dd>
	<dt>29</dt>
		<dd>0.00057169976181281</dd>
	<dt>30</dt>
		<dd>0.00245956350088158</dd>
	<dt>31</dt>
		<dd>0.0521433147841187</dd>
	<dt>32</dt>
		<dd>0.00618519972624196</dd>
	<dt>33</dt>
		<dd>0.0270068624847579</dd>
	<dt>34</dt>
		<dd>0.0546734469026709</dd>
	<dt>35</dt>
		<dd>0.0207164305516394</dd>
	<dt>36</dt>
		<dd>0.0460637815232108</dd>
	<dt>37</dt>
		<dd>0.017653116549724</dd>
	<dt>38</dt>
		<dd>0.0468981687465925</dd>
	<dt>39</dt>
		<dd>0.00231612211992191</dd>
	<dt>40</dt>
		<dd>7.01252986969518e-05</dd>
	<dt>41</dt>
		<dd>2.82782428159267e-05</dd>
	<dt>42</dt>
		<dd>0.0107608311853959</dd>
	<dt>43</dt>
		<dd>0.0239993099563164</dd>
	<dt>44</dt>
		<dd>0.0338106222566802</dd>
	<dt>45</dt>
		<dd>0.0140649780400741</dd>
	<dt>46</dt>
		<dd>0.0180108632138582</dd>
	<dt>47</dt>
		<dd>0.015616989816327</dd>
	<dt>48</dt>
		<dd>0.0041799863723554</dd>
	<dt>49</dt>
		<dd>0.00848351356847245</dd>
	<dt>50</dt>
		<dd>0.00024447866157714</dd>
	<dt>51</dt>
		<dd>0.0368994575992718</dd>
	<dt>52</dt>
		<dd>0.000779447237263226</dd>
	<dt>53</dt>
		<dd>0.00194123452582839</dd>
	<dt>54</dt>
		<dd>0.00244622953945796</dd>
	<dt>55</dt>
		<dd>0.00427036053854858</dd>
	<dt>56</dt>
		<dd>0.00126660899840397</dd>
	<dt>57</dt>
		<dd>0.00182421162616898</dd>
	<dt>58</dt>
		<dd>0.000480470477736417</dd>
	<dt>59</dt>
		<dd>0.0011631813514602</dd>
	<dt>60</dt>
		<dd>0.00318723480542336</dd>
	<dt>61</dt>
		<dd>0.00159551161494257</dd>
	<dt>62</dt>
		<dd>0.000370382579598616</dd>
	<dt>63</dt>
		<dd>0.000157789172046872</dd>
	<dt>64</dt>
		<dd>0.113816531266223</dd>
	<dt>65</dt>
		<dd>0.0282771516630515</dd>
	<dt>66</dt>
		<dd>0.0113937404323357</dd>
	<dt>67</dt>
		<dd>0.00123142188570399</dd>
	<dt>68</dt>
		<dd>0.00226000647914969</dd>
	<dt>69</dt>
		<dd>0.000224132205626816</dd>
	<dt>70</dt>
		<dd>0.0102847893999674</dd>
	<dt>71</dt>
		<dd>0.00284132858478351</dd>
	<dt>72</dt>
		<dd>0.00143122276112942</dd>
	<dt>73</dt>
		<dd>0.000346853791237803</dd>
	<dt>74</dt>
		<dd>0.00294175690728288</dd>
	<dt>75</dt>
		<dd>0.00273824883673405</dd>
	<dt>76</dt>
		<dd>0.000504535673740697</dd>
	<dt>77</dt>
		<dd>0.0138710835097658</dd>
	<dt>78</dt>
		<dd>0.0423096554652718</dd>
	<dt>79</dt>
		<dd>0.00418744026528731</dd>
	<dt>80</dt>
		<dd>0.00386183094222886</dd>
	<dt>81</dt>
		<dd>0.0296582601799113</dd>
	<dt>82</dt>
		<dd>0.000183888843873525</dd>
	<dt>83</dt>
		<dd>0.00214936853690626</dd>
	<dt>84</dt>
		<dd>0.000199392895233863</dd>
	<dt>85</dt>
		<dd>0.091687332458714</dd>
	<dt>86</dt>
		<dd>0.000319899438462304</dd>
	<dt>87</dt>
		<dd>0.000326219222192586</dd>
	<dt>88</dt>
		<dd>4.54738319137531e-05</dd>
	<dt>89</dt>
		<dd>0.000340089305221142</dd>
	<dt>90</dt>
		<dd>0.000788148721884078</dd>
	<dt>91</dt>
		<dd>0.00480120375990711</dd>
	<dt>92</dt>
		<dd>0.00449309454043647</dd>
	<dt>93</dt>
		<dd>0.0118842660416463</dd>
	<dt>94</dt>
		<dd>0.00479636046705399</dd>
	<dt>95</dt>
		<dd>0.0175266588838877</dd>
	<dt>96</dt>
		<dd>0.000173279326820442</dd>
	<dt>97</dt>
		<dd>0.0101230171851761</dd>
	<dt>98</dt>
		<dd>0.0122581830206776</dd>
	<dt>99</dt>
		<dd>0.0239496362196093</dd>
	<dt>100</dt>
		<dd>0.00801050778628487</dd>
</dl>



can have R generate these plots for you:


```R
# plot cooks distance
plot(x=regression.2, which=4)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%2015_files/Learning%20Statistics%20with%20R%20chapter%2015_47_0.png)
    



```R
# plot studentised residual against leverage
plot(x=regression.2, which=5)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%2015_files/Learning%20Statistics%20with%20R%20chapter%2015_48_0.png)
    


What to do about large cook's distance?

try excluding it, but if you do that you need a really good reason why. In general, try to figure out WHY it's so different. 

Try deleting the data from day 64


```R
lm(formula=dan.grump~dan.sleep+baby.sleep, # same formula
    data=parenthood, # same dataframe
    subset=-64, # delete observation 64
  )
```


    
    Call:
    lm(formula = dan.grump ~ dan.sleep + baby.sleep, data = parenthood, 
        subset = -64)
    
    Coefficients:
    (Intercept)    dan.sleep   baby.sleep  
       126.3553      -8.8283      -0.1319  



very little change in the coefficients

## Checking the normality of the residuals


```R
# draw a histogram
hist(x=residuals(regression.2), # data are the residuals
     xlab="Value of residual",
     main="",
     breaks=20
    )
```


    
![png](Learning%20Statistics%20with%20R%20chapter%2015_files/Learning%20Statistics%20with%20R%20chapter%2015_53_0.png)
    


Use Shapiro-wilk test


```R
shapiro.test(residuals(regression.2))
```


    
    	Shapiro-Wilk normality test
    
    data:  residuals(regression.2)
    W = 0.99228, p-value = 0.8414



can also use QQ plot


```R
plot(x=regression.2, which=2)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%2015_files/Learning%20Statistics%20with%20R%20chapter%2015_57_0.png)
    


**Checking the linearity of the relationship**

plot relationship between fitted and observed values


```R
yhat.2 <- fitted.values(object=regression.2)
plot(x=yhat.2,
     y=parenthood$dan.grump,
     xlab="Fitted Values",
     ylab="Observed Values")
```


    
![png](Learning%20Statistics%20with%20R%20chapter%2015_files/Learning%20Statistics%20with%20R%20chapter%2015_59_0.png)
    


As long as this looks somewhat linear, we should be good. 

plot fitted values against residuals


```R
plot(x=regression.2, which=1)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%2015_files/Learning%20Statistics%20with%20R%20chapter%2015_62_0.png)
    


Ideally, the red line is straight and horizontal. 

Can also use `residualPlots` function in car package. 

- also plots results of some curvature tests. For each independent variable, adds the square of that variable and runs a t-test on the b-coefficient. If it is significant, then there might be some nonlinear relationship.


possible solution: transform one of the variables:

Box-Cox transform:
$$
f(x, \lambda) = \frac{x^\lambda-1}{\lambda}
$$

if $\lambda=0$, take natural log


use `boxCox()` in car package


if normalizing, use `powerTransform()` to estimate best value of $\lambda$

**Checking the homogeneity of variance**

plot square root of residuals vs fitted value


```R
plot(x=regression.2, which=3)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%2015_files/Learning%20Statistics%20with%20R%20chapter%2015_65_0.png)
    


use hypothesis test: `ncvTest()` in car (non-constant variance test)

run a regression to see if there's any relationship between residuals and fitted values. 


If homogeneity of variance is violated: standard error with coefficients isnt completely reliable. Try using 'heteroscedasticity corrected covariance matrix'- Sandwich estimators (it gets a little complicated after this)

**Checking for collinearity**

Variance of inflation factors- whether or not the predictors are too highly correlated with each other

$$
\text{VIF}_k = \frac{1}{1-R^2_{(-k)}}
$$

where $-R^2_{(-k)}$ si the R2 value you would get using $X_k$ as the outcome variable and all the other Xs as predictors. The square root of VIF tells you how much wider the confidence interval for the corresponding coefficient is, relative to what you would have expected if the predictors are completely uncorrelated. 


```R
cor(parenthood)
```


<table>
<thead><tr><th></th><th scope=col>dan.sleep</th><th scope=col>baby.sleep</th><th scope=col>dan.grump</th><th scope=col>day</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td> 1.00000000</td><td> 0.62794934</td><td>-0.90338404</td><td>-0.09840768</td></tr>
	<tr><th scope=row>baby.sleep</th><td> 0.62794934</td><td> 1.00000000</td><td>-0.56596373</td><td>-0.01043394</td></tr>
	<tr><th scope=row>dan.grump</th><td>-0.90338404</td><td>-0.56596373</td><td> 1.00000000</td><td> 0.07647926</td></tr>
	<tr><th scope=row>day</th><td>-0.09840768</td><td>-0.01043394</td><td> 0.07647926</td><td> 1.00000000</td></tr>
</tbody>
</table>




```R
regression.3 <- lm(day~baby.sleep+dan.sleep+dan.grump, parenthood)
```

![image.png](attachment:image.png)

## Model Selection

which variables should we include as predictors and which ones should we exclude?

2 principles:
- it's good to have a substantive basis for your choices. Statistics serves the scientific process, not the other way around. 
- As you add new parameters, the model gets better at absorbing random variations (overfitting). (aka Ockham's razon- do not multiply entities beyond necessity)


Akaike information criterion
$$
\text{AIC} = \frac{\text{SS}_{res}}{\hat \sigma^2} + 2K
$$

smaller AIC = better performance

actual value isn't informative- we're interested in the differences- describes how much better one model is than the other


**Backward elimination**

use `step()` function: start with complete model (all predictors) and try all possible ways of removing one predictor. Accept the one with the lowest AIC score



```R
full.model <- lm(formula=dan.grump~dan.sleep+baby.sleep+day, parenthood)
```


```R
step(object=full.model, direction="backward")
```

    Start:  AIC=299.08
    dan.grump ~ dan.sleep + baby.sleep + day
    
                 Df Sum of Sq    RSS    AIC
    - baby.sleep  1       0.1 1837.2 297.08
    - day         1       1.6 1838.7 297.16
    <none>                    1837.1 299.08
    - dan.sleep   1    4909.0 6746.1 427.15
    
    Step:  AIC=297.08
    dan.grump ~ dan.sleep + day
    
                Df Sum of Sq    RSS    AIC
    - day        1       1.6 1838.7 295.17
    <none>                   1837.2 297.08
    - dan.sleep  1    8103.0 9940.1 463.92
    
    Step:  AIC=295.17
    dan.grump ~ dan.sleep
    
                Df Sum of Sq    RSS    AIC
    <none>                   1838.7 295.17
    - dan.sleep  1    8159.9 9998.6 462.50



    
    Call:
    lm(formula = dan.grump ~ dan.sleep, data = parenthood)
    
    Coefficients:
    (Intercept)    dan.sleep  
        125.956       -8.937  



Left column: what change R made to the regression model. \<none\> = no change

**Forward selection**

start with smallest model, only consider possible additions. 

Also need to tell step the largest possible model


```R
null.model <- lm(dan.grump~1, parenthood) # intercept only
step(object=null.model, # start with null
     direction = "forward", # only consider 'addition' moves
     scope = dan.grump~dan.sleep+baby.sleep+day # largest model allowed
    )
```

    Start:  AIC=462.5
    dan.grump ~ 1
    
                 Df Sum of Sq    RSS    AIC
    + dan.sleep   1    8159.9 1838.7 295.17
    + baby.sleep  1    3202.7 6795.9 425.89
    <none>                    9998.6 462.50
    + day         1      58.5 9940.1 463.92
    
    Step:  AIC=295.17
    dan.grump ~ dan.sleep
    
                 Df Sum of Sq    RSS    AIC
    <none>                    1838.7 295.17
    + day         1   1.55760 1837.2 297.08
    + baby.sleep  1   0.02858 1838.7 297.16



    
    Call:
    lm(formula = dan.grump ~ dan.sleep, data = parenthood)
    
    Coefficients:
    (Intercept)    dan.sleep  
        125.956       -8.937  



Foudn the same model, although they don't always start and end in the same place. 

**A caveat**


There are many... many tools available for model selection. Trust your instincts. Interpretability matters.

**Comparing two regression models**

Does the amout of sleep that baby got have any relationship to dan's grumpiness? Also, make sure that day measured has no effect. 

check relationship between baby.sleep and dan.grump from the perspective that dan.grump adn day are nuisance variables (i.e. **covariates**) that we want to control for. 

Is `dan.grump ~ dan.sleep + day + baby.sleep` (M1) a better regression model than `dan.grump~dan.sleep+day` (M0)


AIC approach


```R
M0 <- lm(dan.grump~dan.sleep+day, parenthood)
M1 <- lm(dan.grump~dan.sleep+day+baby.sleep, parenthood)

AIC(M0, M1)
```


<table>
<thead><tr><th></th><th scope=col>df</th><th scope=col>AIC</th></tr></thead>
<tbody>
	<tr><th scope=row>M0</th><td>4       </td><td>582.8681</td></tr>
	<tr><th scope=row>M1</th><td>5       </td><td>584.8646</td></tr>
</tbody>
</table>



Therefore M0 is better

Also think about hypothesis testing framework

M0 contains a subset of the predictors from M1. 

M0 is nested within M1, or M0 is a submodel of M1.

This means M0 is a null hypothesis and M1 is an alternative hypothesis. 

Fit each model and get residual sum of squares for each - then do an F-test

![image.png](attachment:image.png)

N = number of observations, p = number of predictors, k = difference in number of parameters between the two models. dof = k and N-p-1.  


```R
print(anova(M0, M1))
```

    Analysis of Variance Table
    
    Model 1: dan.grump ~ dan.sleep + day
    Model 2: dan.grump ~ dan.sleep + day + baby.sleep
      Res.Df    RSS Df Sum of Sq      F Pr(>F)
    1     97 1837.2                           
    2     96 1837.1  1  0.063688 0.0033 0.9541


Since p>0.05, retain the null

Hierarchical regression: add all covariates into a null model, add variables of interest into alternative model, then compare the two models in hypothesis testing framework
