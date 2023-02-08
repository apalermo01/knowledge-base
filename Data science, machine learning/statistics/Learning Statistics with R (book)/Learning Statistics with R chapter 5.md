# Descriptive Statistics


```R
load("aflsmall.Rdata")
library(lsr)
who()
```


       -- Name --      -- Class --   -- Size --
       afl.finalists   factor        400       
       afl.margins     numeric       176       



```R
print(afl.margins)
```

      [1]  56  31  56   8  32  14  36  56  19   1   3 104  43  44  72   9  28  25
     [19]  27  55  20  16  16   7  23  40  48  64  22  55  95  15  49  52  50  10
     [37]  65  12  39  36   3  26  23  20  43 108  53  38   4   8   3  13  66  67
     [55]  50  61  36  38  29   9  81   3  26  12  36  37  70   1  35  12  50  35
     [73]   9  54  47   8  47   2  29  61  38  41  23  24   1   9  11  10  29  47
     [91]  71  38  49  65  18   0  16   9  19  36  60  24  25  44  55   3  57  83
    [109]  84  35   4  35  26  22   2  14  19  30  19  68  11  75  48  32  36  39
    [127]  50  11   0  63  82  26   3  82  73  19  33  48   8  10  53  20  71  75
    [145]  76  54  44   5  22  94  29   8  98   9  89   1 101   7  21  52  42  21
    [163] 116   3  44  29  27  16   6  44   3  28  38  29  10  10



```R
hist(afl.margins)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%205_3_0.png)
    


## Measures of central tendency

**mean**


```R
mean(afl.margins)
```


35.3011363636364


sort values


```R
sort(afl.margins)
```


<ol class=list-inline>
	<li>0</li>
	<li>0</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>4</li>
	<li>4</li>
	<li>5</li>
	<li>6</li>
	<li>7</li>
	<li>7</li>
	<li>8</li>
	<li>8</li>
	<li>8</li>
	<li>8</li>
	<li>8</li>
	<li>9</li>
	<li>9</li>
	<li>9</li>
	<li>9</li>
	<li>9</li>
	<li>9</li>
	<li>10</li>
	<li>10</li>
	<li>10</li>
	<li>10</li>
	<li>10</li>
	<li>11</li>
	<li>11</li>
	<li>11</li>
	<li>12</li>
	<li>12</li>
	<li>12</li>
	<li>13</li>
	<li>14</li>
	<li>14</li>
	<li>15</li>
	<li>16</li>
	<li>16</li>
	<li>16</li>
	<li>16</li>
	<li>18</li>
	<li>19</li>
	<li>19</li>
	<li>19</li>
	<li>19</li>
	<li>19</li>
	<li>20</li>
	<li>20</li>
	<li>20</li>
	<li>21</li>
	<li>21</li>
	<li>22</li>
	<li>22</li>
	<li>22</li>
	<li>23</li>
	<li>23</li>
	<li>23</li>
	<li>24</li>
	<li>24</li>
	<li>25</li>
	<li>25</li>
	<li>26</li>
	<li>26</li>
	<li>26</li>
	<li>26</li>
	<li>27</li>
	<li>27</li>
	<li>28</li>
	<li>28</li>
	<li>29</li>
	<li>29</li>
	<li>29</li>
	<li>29</li>
	<li>29</li>
	<li>29</li>
	<li>30</li>
	<li>31</li>
	<li>32</li>
	<li>32</li>
	<li>33</li>
	<li>35</li>
	<li>35</li>
	<li>35</li>
	<li>35</li>
	<li>36</li>
	<li>36</li>
	<li>36</li>
	<li>36</li>
	<li>36</li>
	<li>36</li>
	<li>37</li>
	<li>38</li>
	<li>38</li>
	<li>38</li>
	<li>38</li>
	<li>38</li>
	<li>39</li>
	<li>39</li>
	<li>40</li>
	<li>41</li>
	<li>42</li>
	<li>43</li>
	<li>43</li>
	<li>44</li>
	<li>44</li>
	<li>44</li>
	<li>44</li>
	<li>44</li>
	<li>47</li>
	<li>47</li>
	<li>47</li>
	<li>48</li>
	<li>48</li>
	<li>48</li>
	<li>49</li>
	<li>49</li>
	<li>50</li>
	<li>50</li>
	<li>50</li>
	<li>50</li>
	<li>52</li>
	<li>52</li>
	<li>53</li>
	<li>53</li>
	<li>54</li>
	<li>54</li>
	<li>55</li>
	<li>55</li>
	<li>55</li>
	<li>56</li>
	<li>56</li>
	<li>56</li>
	<li>57</li>
	<li>60</li>
	<li>61</li>
	<li>61</li>
	<li>63</li>
	<li>64</li>
	<li>65</li>
	<li>65</li>
	<li>66</li>
	<li>67</li>
	<li>68</li>
	<li>70</li>
	<li>71</li>
	<li>71</li>
	<li>72</li>
	<li>73</li>
	<li>75</li>
	<li>75</li>
	<li>76</li>
	<li>81</li>
	<li>82</li>
	<li>82</li>
	<li>83</li>
	<li>84</li>
	<li>89</li>
	<li>94</li>
	<li>95</li>
	<li>98</li>
	<li>101</li>
	<li>104</li>
	<li>108</li>
	<li>116</li>
</ol>



**median**


```R
median(afl.margins)
```


30.5


**mean or median?**

mean = center of mass of data<br>
median = middle observation<br>

if nominal scale- don't use median or mean<br>
if ordinal scale- you most likely want median<br>
if interval / ratio scale- either one, depends on exactly what you're trying to do. 

Systematic differences between mean and median:<br>
mean is much more sensitive to outliers<br>
mean -> big picture<br> 
median -> what's typical<br>

**example**<br>
www.abc.net.au/news/stories/2010/09/24/3021480.htm

article criticized a bank for calculating house price to income ratio by "comparing average incomes with median housing prices", which lowered the ratio

**trimmed mean**

discard the n% largest and smallest observations<br>
0% trimmed mean = normal mean<br>
50% trimmed mean = median


```R
dataset <- c(-15, 2, 3, 4, 5, 6, 7, 8, 9, 12)
```


```R
mean(dataset)
```


4.1



```R
median(dataset)
```


5.5



```R
mean(dataset, trim=0.1)
```


5.5



```R
mean(afl.margins)
```


35.3011363636364



```R
mean(afl.margins, trim=0.05)
```


33.75


**mode**


```R
print(afl.finalists)
```

      [1] Hawthorn         Melbourne        Carlton          Melbourne       
      [5] Hawthorn         Carlton          Melbourne        Carlton         
      [9] Hawthorn         Melbourne        Melbourne        Hawthorn        
     [13] Melbourne        Essendon         Hawthorn         Geelong         
     [17] Geelong          Hawthorn         Collingwood      Melbourne       
     [21] Collingwood      West Coast       Collingwood      Essendon        
     [25] Collingwood      Melbourne        Hawthorn         Geelong         
     [29] Hawthorn         West Coast       West Coast       Hawthorn        
     [33] St Kilda         West Coast       Geelong          Western Bulldogs
     [37] West Coast       Geelong          West Coast       West Coast      
     [41] Carlton          Adelaide         Carlton          Essendon        
     [45] Essendon         Essendon         North Melbourne  Geelong         
     [49] West Coast       Melbourne        Geelong          Melbourne       
     [53] West Coast       Geelong          West Coast       North Melbourne 
     [57] Essendon         Geelong          Carlton          Richmond        
     [61] North Melbourne  Carlton          Geelong          Carlton         
     [65] Brisbane         West Coast       Sydney           North Melbourne 
     [69] Essendon         Brisbane         Sydney           North Melbourne 
     [73] North Melbourne  Western Bulldogs St Kilda         North Melbourne 
     [77] Adelaide         North Melbourne  Adelaide         St Kilda        
     [81] Adelaide         Adelaide         North Melbourne  Sydney          
     [85] Melbourne        Western Bulldogs Melbourne        Adelaide        
     [89] North Melbourne  Adelaide         Adelaide         West Coast      
     [93] North Melbourne  Brisbane         Essendon         Brisbane        
     [97] Carlton          North Melbourne  Carlton          North Melbourne 
    [101] Geelong          Essendon         Brisbane         Carlton         
    [105] North Melbourne  Carlton          Melbourne        Essendon        
    [109] Essendon         Essendon         Carlton          Brisbane        
    [113] Hawthorn         Richmond         Port Adelaide    Essendon        
    [117] Brisbane         Essendon         Port Adelaide    Brisbane        
    [121] Essendon         Melbourne        Port Adelaide    Adelaide        
    [125] Collingwood      Brisbane         Brisbane         Fremantle       
    [129] Collingwood      Adelaide         Port Adelaide    Brisbane        
    [133] Port Adelaide    Collingwood      Sydney           Collingwood     
    [137] Brisbane         Melbourne        Sydney           Port Adelaide   
    [141] St Kilda         Geelong          Port Adelaide    Brisbane        
    [145] Port Adelaide    West Coast       Geelong          Adelaide        
    [149] North Melbourne  Sydney           Adelaide         St Kilda        
    [153] West Coast       Sydney           St Kilda         Adelaide        
    [157] West Coast       Collingwood      Fremantle        West Coast      
    [161] Sydney           Adelaide         Sydney           Port Adelaide   
    [165] Hawthorn         Collingwood      Geelong          West Coast      
    [169] North Melbourne  Geelong          Port Adelaide    Geelong         
    [173] Hawthorn         Adelaide         Sydney           Geelong         
    [177] Western Bulldogs St Kilda         Geelong          Hawthorn        
    [181] Geelong          Adelaide         Geelong          Brisbane        
    [185] St Kilda         Western Bulldogs Collingwood      St Kilda        
    [189] Geelong          St Kilda         Geelong          Fremantle       
    [193] Collingwood      Sydney           Geelong          Western Bulldogs
    [197] Collingwood      St Kilda         Collingwood      Collingwood     
    [201] Sydney           North Melbourne  Hawthorn         Sydney          
    [205] Melbourne        Hawthorn         West Coast       Collingwood     
    [209] Carlton          Collingwood      Carlton          Melbourne       
    [213] Collingwood      Geelong          Essendon         Melbourne       
    [217] Essendon         Geelong          West Coast       Hawthorn        
    [221] West Coast       Melbourne        Essendon         West Coast      
    [225] Essendon         Essendon         West Coast       St Kilda        
    [229] Geelong          Melbourne        Geelong          West Coast      
    [233] Collingwood      Hawthorn         Western Bulldogs St Kilda        
    [237] Geelong          Western Bulldogs Geelong          North Melbourne 
    [241] Essendon         Hawthorn         Adelaide         West Coast      
    [245] Adelaide         Carlton          Hawthorn         Western Bulldogs
    [249] Collingwood      Carlton          Carlton          Western Bulldogs
    [253] Melbourne        North Melbourne  Geelong          Richmond        
    [257] West Coast       Western Bulldogs Brisbane         Essendon        
    [261] West Coast       North Melbourne  Richmond         Geelong         
    [265] Essendon         Carlton          Hawthorn         Geelong         
    [269] West Coast       Carlton          Essendon         Brisbane        
    [273] Sydney           Sydney           Brisbane         Geelong         
    [277] West Coast       West Coast       Geelong          North Melbourne 
    [281] Western Bulldogs St Kilda         Essendon         St Kilda        
    [285] Adelaide         West Coast       St Kilda         Sydney          
    [289] Melbourne        Western Bulldogs North Melbourne  Western Bulldogs
    [293] Port Adelaide    Carlton          Sydney           Western Bulldogs
    [297] West Coast       Brisbane         Essendon         Carlton         
    [301] Hawthorn         North Melbourne  Western Bulldogs Melbourne       
    [305] Hawthorn         Brisbane         North Melbourne  Carlton         
    [309] Melbourne        Richmond         Adelaide         Port Adelaide   
    [313] Sydney           Carlton          Hawthorn         Hawthorn        
    [317] Richmond         Brisbane         Collingwood      Adelaide        
    [321] West Coast       North Melbourne  Essendon         Melbourne       
    [325] Adelaide         Port Adelaide    Collingwood      Essendon        
    [329] Brisbane         West Coast       Sydney           Adelaide        
    [333] Essendon         Port Adelaide    Brisbane         Brisbane        
    [337] St Kilda         Essendon         West Coast       Geelong         
    [341] Sydney           Essendon         St Kilda         Geelong         
    [345] Brisbane         Sydney           Melbourne        St Kilda        
    [349] Port Adelaide    Geelong          Port Adelaide    Sydney          
    [353] Adelaide         West Coast       Melbourne        Fremantle       
    [357] Sydney           Western Bulldogs Melbourne        Western Bulldogs
    [361] Fremantle        West Coast       West Coast       West Coast      
    [365] Adelaide         Sydney           North Melbourne  Collingwood     
    [369] Hawthorn         Collingwood      North Melbourne  Port Adelaide   
    [373] Western Bulldogs Collingwood      North Melbourne  St Kilda        
    [377] Sydney           Collingwood      Western Bulldogs St Kilda        
    [381] Hawthorn         Essendon         Western Bulldogs Carlton         
    [385] Collingwood      Brisbane         Adelaide         Western Bulldogs
    [389] Collingwood      Geelong          St Kilda         Hawthorn        
    [393] Western Bulldogs Carlton          Fremantle        Sydney          
    [397] Geelong          Western Bulldogs St Kilda         St Kilda        
    17 Levels: Adelaide Brisbane Carlton Collingwood Essendon Fitzroy ... Western Bulldogs


make a frequency table


```R
table(afl.finalists)
```


    afl.finalists
            Adelaide         Brisbane          Carlton      Collingwood 
                  26               25               26               28 
            Essendon          Fitzroy        Fremantle          Geelong 
                  32                0                6               39 
            Hawthorn        Melbourne  North Melbourne    Port Adelaide 
                  27               28               28               17 
            Richmond         St Kilda           Sydney       West Coast 
                   6               24               26               38 
    Western Bulldogs 
                  24 


no native function in R for calculating mode, but there is one in the lsr package


```R
modeOf(afl.finalists)
```


'Geelong'


also maximum frequency


```R
maxFreq(afl.finalists)
```


39


## Measures of variability

**range**<br>
max - min


```R
range(afl.margins)
```


<ol class=list-inline>
	<li>0</li>
	<li>116</li>
</ol>



**interquartile range**


```R
quantile(afl.margins, probs=0.5)
```


<strong>50%:</strong> 30.5



```R
quantile(afl.margins, probs=c(0.25, 0.75))
```


<dl class=dl-horizontal>
	<dt>25%</dt>
		<dd>12.75</dd>
	<dt>75%</dt>
		<dd>50.5</dd>
</dl>




```R
IQR(afl.margins)
```


37.75


**mean absolute deviation**

select mean or median, then report the mean or median of the deviations from that measure<br>
(mean or median) absolute deviation from the (mean or median) (4 options)

average absolute deviation:
$$
\text{AAD}(X) = \frac{1}{N} \sum_{i=1}^{N} |X_i - \bar{X}|
$$

where $\bar{X}$ can be the mean or median

use aad function in the lsr package


```R
aad(afl.margins)
```


21.1012396694215


**Variance**

use squared deviation instead of absolute deviation


```R
var(afl.margins)
```


679.834512987013



```R
mean( (afl.margins - mean(afl.margins))^2)
```


675.971816890496


why are these different?

var(afl.margins):
$$
\frac{1}{N-1} \sum_{i=1}^{N} (X_i - \bar{X})^2
$$

mean( (afl.margins - mean(afl.margins))^2):
$$
\frac{1}{N} \sum_{i=1}^{N} (X_i - \bar{X})^2
$$

Technical details will be reserved for chapter 10. Related to sample statistics vs. population parameters

**standard deviation**

represent variance in units of the original measurement (i.e. not squared)

Root mean squared deviation = RMSD
$$
s = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (X_i - \bar{X})^2}
$$

$$
\hat{\sigma} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (X_i - \bar{X})^2}
$$


```R
sd(afl.margins)
```


26.0736363591083


## Median absolute deviation


```R
# mean absolute deviation from the mean
print(mean(abs(afl.margins - mean(afl.margins))))

# median absolute deviation from the median
print(median(abs(afl.margins - median(afl.margins))))
```

    [1] 21.10124
    [1] 19.5


use builtin function mad()


```R
mad(afl.margins, constant=1)
```


19.5


the MAD is sometimes used as a robust standard deviation (i.e. less prone to outliers), which is where the constant comes in. By default, constant = 1.4826, which makes the MAD directly comparable to the standard deviation


```R
mad(afl.margins)
```


28.9107


### Summary

- rnage - full spread
- interquartile range - where the middle half sits
- mean absolute deviation - how far observations are "on average" from the mean. Rarely used.
- variance - usually correct way to measure variation around mean
- standard deviation - square root of variance- default measure of variation
- median absolute deviation - median deviation from median. Corrected form = robust way to get std

IQR and standard deviation are most common

## Skew and kurtosis

**skewness** = measure of asymmetry

$$
\text{skewness}(X) = \frac{1}{N \hat{\sigma}^3} \sum_{i=1}^{N} (X_i - \bar{X})^3
$$

use psych library for skey


```R
library(psych)
skew(afl.margins)
```


0.767155515761526


**kurtosis** = pointiness of a dataset
![image.png](attachment:image.png)

$$
\text{kurtosis}(X) = \frac{1}{N \hat{\sigma}^4} \sum_{i=1}^{N} (X_i - \bar{X})^4 - 3
$$

![image-2.png](attachment:image-2.png)

use psych package for kurtosis


```R
kurtosi(afl.margins)
```


0.0296263303725643


## Summary of a variable

summary() = generic builtin function


```R
summary(afl.margins)
```


       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       0.00   12.75   30.50   35.30   50.50  116.00 



```R
blowouts <- afl.margins > 50
summary(blowouts)
```


       Mode   FALSE    TRUE 
    logical     132      44 



```R
summary(afl.finalists)
```


<dl class=dl-horizontal>
	<dt>Adelaide</dt>
		<dd>26</dd>
	<dt>Brisbane</dt>
		<dd>25</dd>
	<dt>Carlton</dt>
		<dd>26</dd>
	<dt>Collingwood</dt>
		<dd>28</dd>
	<dt>Essendon</dt>
		<dd>32</dd>
	<dt>Fitzroy</dt>
		<dd>0</dd>
	<dt>Fremantle</dt>
		<dd>6</dd>
	<dt>Geelong</dt>
		<dd>39</dd>
	<dt>Hawthorn</dt>
		<dd>27</dd>
	<dt>Melbourne</dt>
		<dd>28</dd>
	<dt>North Melbourne</dt>
		<dd>28</dd>
	<dt>Port Adelaide</dt>
		<dd>17</dd>
	<dt>Richmond</dt>
		<dd>6</dd>
	<dt>St Kilda</dt>
		<dd>24</dd>
	<dt>Sydney</dt>
		<dd>26</dd>
	<dt>West Coast</dt>
		<dd>38</dd>
	<dt>Western Bulldogs</dt>
		<dd>24</dd>
</dl>




```R
f2 <- as.character(afl.finalists)
summary(f2)
```


       Length     Class      Mode 
          400 character character 


### Summarizing a data frame


```R
load("clinicaltrial.Rdata")
who(TRUE)
```


       -- Name --      -- Class --   -- Size --
       afl.finalists   factor        400       
       afl.margins     numeric       176       
       blowouts        logical       176       
       clin.trial      data.frame    18 x 3    
        $drug          factor        18        
        $therapy       factor        18        
        $mood.gain     numeric       18        
       dataset         numeric       10        
       f2              character     400       



```R
summary(clin.trial)
```


           drug         therapy    mood.gain     
     placebo :6   no.therapy:9   Min.   :0.1000  
     anxifree:6   CBT       :9   1st Qu.:0.4250  
     joyzepam:6                  Median :0.8500  
                                 Mean   :0.8833  
                                 3rd Qu.:1.3000  
                                 Max.   :1.8000  


### Describing a data frame

describe() in psych package, for interval or ratio scale


```R
describe(clin.trial)
```


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>drug*</th><td>1        </td><td>18       </td><td>2.0000000</td><td>0.8401681</td><td>2.00     </td><td>2.000    </td><td>1.48260  </td><td>1.0      </td><td>3.0      </td><td>2.0      </td><td>0.0000000</td><td>-1.662037</td><td>0.1980295</td></tr>
	<tr><th scope=row>therapy*</th><td>2        </td><td>18       </td><td>1.5000000</td><td>0.5144958</td><td>1.50     </td><td>1.500    </td><td>0.74130  </td><td>1.0      </td><td>2.0      </td><td>1.0      </td><td>0.0000000</td><td>-2.108025</td><td>0.1212678</td></tr>
	<tr><th scope=row>mood.gain</th><td>3        </td><td>18       </td><td>0.8833333</td><td>0.5338539</td><td>0.85     </td><td>0.875    </td><td>0.66717  </td><td>0.1      </td><td>1.8      </td><td>1.7      </td><td>0.1333981</td><td>-1.439739</td><td>0.1258306</td></tr>
</tbody>
</table>



## Descriptive statistics for each group

by(), describeBy(), and aggregate()


```R
describeBy(clin.trial, group=clin.trial$therapy)
```


    
     Descriptive statistics by group 
    group: no.therapy
              vars n mean   sd median trimmed  mad min max range skew kurtosis   se
    drug*        1 9 2.00 0.87    2.0    2.00 1.48 1.0 3.0   2.0 0.00    -1.81 0.29
    therapy*     2 9 1.00 0.00    1.0    1.00 0.00 1.0 1.0   0.0  NaN      NaN 0.00
    mood.gain    3 9 0.72 0.59    0.5    0.72 0.44 0.1 1.7   1.6 0.51    -1.59 0.20
    ------------------------------------------------------------ 
    group: CBT
              vars n mean   sd median trimmed  mad min max range  skew kurtosis
    drug*        1 9 2.00 0.87    2.0    2.00 1.48 1.0 3.0   2.0  0.00    -1.81
    therapy*     2 9 2.00 0.00    2.0    2.00 0.00 2.0 2.0   0.0   NaN      NaN
    mood.gain    3 9 1.04 0.45    1.1    1.04 0.44 0.3 1.8   1.5 -0.03    -1.12
                se
    drug*     0.29
    therapy*  0.00
    mood.gain 0.15


by() just applied some function to each group


```R
by(data=clin.trial, INDICES=clin.trial$therapy, FUN=describe)
```


    clin.trial$therapy: no.therapy
              vars n mean   sd median trimmed  mad min max range skew kurtosis   se
    drug*        1 9 2.00 0.87    2.0    2.00 1.48 1.0 3.0   2.0 0.00    -1.81 0.29
    therapy*     2 9 1.00 0.00    1.0    1.00 0.00 1.0 1.0   0.0  NaN      NaN 0.00
    mood.gain    3 9 0.72 0.59    0.5    0.72 0.44 0.1 1.7   1.6 0.51    -1.59 0.20
    ------------------------------------------------------------ 
    clin.trial$therapy: CBT
              vars n mean   sd median trimmed  mad min max range  skew kurtosis
    drug*        1 9 2.00 0.87    2.0    2.00 1.48 1.0 3.0   2.0  0.00    -1.81
    therapy*     2 9 2.00 0.00    2.0    2.00 0.00 2.0 2.0   0.0   NaN      NaN
    mood.gain    3 9 1.04 0.45    1.1    1.04 0.44 0.3 1.8   1.5 -0.03    -1.12
                se
    drug*     0.29
    therapy*  0.00
    mood.gain 0.15



```R
by(data=clin.trial, INDICES=clin.trial$therapy, FUN=summary)
```


    clin.trial$therapy: no.therapy
           drug         therapy    mood.gain     
     placebo :3   no.therapy:9   Min.   :0.1000  
     anxifree:3   CBT       :0   1st Qu.:0.3000  
     joyzepam:3                  Median :0.5000  
                                 Mean   :0.7222  
                                 3rd Qu.:1.3000  
                                 Max.   :1.7000  
    ------------------------------------------------------------ 
    clin.trial$therapy: CBT
           drug         therapy    mood.gain    
     placebo :3   no.therapy:0   Min.   :0.300  
     anxifree:3   CBT       :9   1st Qu.:0.800  
     joyzepam:3                  Median :1.100  
                                 Mean   :1.044  
                                 3rd Qu.:1.300  
                                 Max.   :1.800  


**aggregate**

formula = which variable to analyze and which variables used to specify the groups<br>
ex: look at mood.gain for each combination of drug and therapy, use `mood.gain ~ drug + therapy`

data = data frame

FUN = function


```R
aggregate(formula = mood.gain~drug+therapy,
          data = clin.trial,
          FUN=mean)
```


<table>
<thead><tr><th scope=col>drug</th><th scope=col>therapy</th><th scope=col>mood.gain</th></tr></thead>
<tbody>
	<tr><td>placebo   </td><td>no.therapy</td><td>0.300000  </td></tr>
	<tr><td>anxifree  </td><td>no.therapy</td><td>0.400000  </td></tr>
	<tr><td>joyzepam  </td><td>no.therapy</td><td>1.466667  </td></tr>
	<tr><td>placebo   </td><td>CBT       </td><td>0.600000  </td></tr>
	<tr><td>anxifree  </td><td>CBT       </td><td>1.033333  </td></tr>
	<tr><td>joyzepam  </td><td>CBT       </td><td>1.500000  </td></tr>
</tbody>
</table>




```R
aggregate(formula = mood.gain~drug+therapy,
          data = clin.trial,
          FUN=sd)
```


<table>
<thead><tr><th scope=col>drug</th><th scope=col>therapy</th><th scope=col>mood.gain</th></tr></thead>
<tbody>
	<tr><td>placebo   </td><td>no.therapy</td><td>0.2000000 </td></tr>
	<tr><td>anxifree  </td><td>no.therapy</td><td>0.2000000 </td></tr>
	<tr><td>joyzepam  </td><td>no.therapy</td><td>0.2081666 </td></tr>
	<tr><td>placebo   </td><td>CBT       </td><td>0.3000000 </td></tr>
	<tr><td>anxifree  </td><td>CBT       </td><td>0.2081666 </td></tr>
	<tr><td>joyzepam  </td><td>CBT       </td><td>0.2645751 </td></tr>
</tbody>
</table>



## Standard Scores

this is another name for z-score

$$
z_i = \frac{X_i - \bar{X}}{\hat{\sigma}}
$$

pnorm - converts standard score to percentile


```R
pnorm(3.6)
```


0.999840891409842


## Correlations


```R
load("parenthood.Rdata")
who(TRUE)
```


       -- Name --      -- Class --   -- Size --
       afl.finalists   factor        400       
       afl.margins     numeric       176       
       blowouts        logical       176       
       clin.trial      data.frame    18 x 3    
        $drug          factor        18        
        $therapy       factor        18        
        $mood.gain     numeric       18        
       dataset         numeric       10        
       f2              character     400       
       parenthood      data.frame    100 x 4   
        $dan.sleep     numeric       100       
        $baby.sleep    numeric       100       
        $dan.grump     numeric       100       
        $day           integer       100       



```R
describe(parenthood)
```


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td>1          </td><td>100        </td><td> 6.9652    </td><td> 1.015884  </td><td> 7.03      </td><td> 7.003125  </td><td> 1.089711  </td><td> 4.84      </td><td>  9.00     </td><td> 4.16      </td><td>-0.28683284</td><td>-0.7225107 </td><td>0.1015884  </td></tr>
	<tr><th scope=row>baby.sleep</th><td>2          </td><td>100        </td><td> 8.0492    </td><td> 2.074232  </td><td> 7.95      </td><td> 8.047750  </td><td> 2.327682  </td><td> 3.25      </td><td> 12.07     </td><td> 8.82      </td><td>-0.02310118</td><td>-0.6893588 </td><td>0.2074232  </td></tr>
	<tr><th scope=row>dan.grump</th><td>3          </td><td>100        </td><td>63.7100    </td><td>10.049670  </td><td>62.00      </td><td>63.162500  </td><td> 9.636900  </td><td>41.00      </td><td> 91.00     </td><td>50.00      </td><td> 0.43410145</td><td>-0.1576624 </td><td>1.0049670  </td></tr>
	<tr><th scope=row>day</th><td>4          </td><td>100        </td><td>50.5000    </td><td>29.011492  </td><td>50.50      </td><td>50.500000  </td><td>37.065000  </td><td> 1.00      </td><td>100.00     </td><td>99.00      </td><td> 0.00000000</td><td>-1.2360552 </td><td>2.9011492  </td></tr>
</tbody>
</table>



### Correlation coefficient

**Covariance**
$$
\text{Cov}(X, Y) = \frac{1}{N-1} \sum_{i=1}^{N} (X_i - \bar{X})(Y_i - \bar{Y})
$$

**pearson's correlation coefficient**

$$
r_{XY} = \frac{\text{Cov}(X, Y)}{\hat{\sigma}_X \hat{\sigma}_Y}
$$



```R
cor(x=parenthood$dan.sleep, y=parenthood$dan.grump)
```


-0.903384037465727


correlation matrix


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



Interpreting the correlation coefficient is highly context dependent

"Anscombe's Quartet"

![image.png](attachment:image.png)

**Spearman's rank correlations**

correlation for nonlinear relationships


```R
load("effort.Rdata")
who(TRUE)
```


       -- Name --      -- Class --   -- Size --
       afl.finalists   factor        400       
       afl.margins     numeric       176       
       blowouts        logical       176       
       clin.trial      data.frame    18 x 3    
        $drug          factor        18        
        $therapy       factor        18        
        $mood.gain     numeric       18        
       dataset         numeric       10        
       effort          data.frame    10 x 2    
        $hours         numeric       10        
        $grade         numeric       10        
       f2              character     400       
       parenthood      data.frame    100 x 4   
        $dan.sleep     numeric       100       
        $baby.sleep    numeric       100       
        $dan.grump     numeric       100       
        $day           integer       100       



```R
cor(effort$hours, effort$grade)
```


0.909401965861253



```R
hours.rank <- rank(effort$hours)
grade.rank <- rank(effort$grade)

cor(hours.rank, grade.rank)
```


1


this is basically spearman's rank order correlation


```R
cor(effort$hours, effort$grade, method="spearman")
```


1


### correlate() function


```R
load("work.Rdata")
who(TRUE)
```


       -- Name --      -- Class --   -- Size --
       afl.finalists   factor        400       
       afl.margins     numeric       176       
       blowouts        logical       176       
       clin.trial      data.frame    18 x 3    
        $drug          factor        18        
        $therapy       factor        18        
        $mood.gain     numeric       18        
       dataset         numeric       10        
       effort          data.frame    10 x 2    
        $hours         numeric       10        
        $grade         numeric       10        
       f2              character     400       
       grade.rank      numeric       10        
       hours.rank      numeric       10        
       parenthood      data.frame    100 x 4   
        $dan.sleep     numeric       100       
        $baby.sleep    numeric       100       
        $dan.grump     numeric       100       
        $day           integer       100       
       work            data.frame    49 x 7    
        $hours         numeric       49        
        $tasks         numeric       49        
        $pay           numeric       49        
        $day           integer       49        
        $weekday       factor        49        
        $week          numeric       49        
        $day.type      factor        49        



```R
head(work)
```


<table>
<thead><tr><th scope=col>hours</th><th scope=col>tasks</th><th scope=col>pay</th><th scope=col>day</th><th scope=col>weekday</th><th scope=col>week</th><th scope=col>day.type</th></tr></thead>
<tbody>
	<tr><td>7.2      </td><td>14       </td><td>41       </td><td>1        </td><td>Tuesday  </td><td>1        </td><td>weekday  </td></tr>
	<tr><td>7.4      </td><td>11       </td><td>39       </td><td>2        </td><td>Wednesday</td><td>1        </td><td>weekday  </td></tr>
	<tr><td>6.6      </td><td>14       </td><td>13       </td><td>3        </td><td>Thursday </td><td>1        </td><td>weekday  </td></tr>
	<tr><td>6.5      </td><td>22       </td><td>47       </td><td>4        </td><td>Friday   </td><td>1        </td><td>weekday  </td></tr>
	<tr><td>3.1      </td><td> 5       </td><td> 4       </td><td>5        </td><td>Saturday </td><td>1        </td><td>weekend  </td></tr>
	<tr><td>3.0      </td><td> 7       </td><td>12       </td><td>6        </td><td>Sunday   </td><td>1        </td><td>weekend  </td></tr>
</tbody>
</table>




```R
cor(work)
```


    Error in cor(work): 'x' must be numeric
    Traceback:


    1. cor(work)

    2. stop("'x' must be numeric")



```R
print(correlate(work))
```

    
    CORRELATIONS
    ============
    - correlation type:  pearson 
    - correlations shown only when both variables are numeric
    
              hours  tasks   pay    day weekday   week day.type
    hours         .  0.800 0.760 -0.049       .  0.018        .
    tasks     0.800      . 0.720 -0.072       . -0.013        .
    pay       0.760  0.720     .  0.137       .  0.196        .
    day      -0.049 -0.072 0.137      .       .  0.990        .
    weekday       .      .     .      .       .      .        .
    week      0.018 -0.013 0.196  0.990       .      .        .
    day.type      .      .     .      .       .      .        .


### Handling missing values

**single variable case**"


```R
partial <- c(10, 20, NA, 30)
```


```R
mean(partial)
```


&lt;NA&gt;


include optional arg: na.rm -> "remove NA values"


```R
mean(partial, na.rm=TRUE)
```


20


In all functions, na.rm will completely ingore missing values (i.e. in the above example N=3, not 4)

**missing values in pairwise calculations**


```R
load("parenthood2.Rdata")
print(head(parenthood2))
```

      dan.sleep baby.sleep dan.grump day
    1      7.59         NA        56   1
    2      7.91      11.66        60   2
    3      5.14       7.92        82   3
    4      7.71       9.61        55   4
    5      6.68       9.75        NA   5
    6      5.99       5.04        72   6



```R
describe(parenthood2)
```


<table>
<thead><tr><th></th><th scope=col>vars</th><th scope=col>n</th><th scope=col>mean</th><th scope=col>sd</th><th scope=col>median</th><th scope=col>trimmed</th><th scope=col>mad</th><th scope=col>min</th><th scope=col>max</th><th scope=col>range</th><th scope=col>skew</th><th scope=col>kurtosis</th><th scope=col>se</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td>1          </td><td> 91        </td><td> 6.976923  </td><td> 1.020409  </td><td> 7.03      </td><td> 7.022740  </td><td> 1.126776  </td><td> 4.84      </td><td>  9.00     </td><td> 4.16      </td><td>-0.33466179</td><td>-0.7278902 </td><td>0.1069679  </td></tr>
	<tr><th scope=row>baby.sleep</th><td>2          </td><td> 89        </td><td> 8.114494  </td><td> 2.046821  </td><td> 8.20      </td><td> 8.129452  </td><td> 2.283204  </td><td> 3.25      </td><td> 12.07     </td><td> 8.82      </td><td>-0.09287191</td><td>-0.5943754 </td><td>0.2169626  </td></tr>
	<tr><th scope=row>dan.grump</th><td>3          </td><td> 92        </td><td>63.152174  </td><td> 9.851574  </td><td>61.00      </td><td>62.662162  </td><td>10.378200  </td><td>41.00      </td><td> 89.00     </td><td>48.00      </td><td> 0.38335860</td><td>-0.3144273 </td><td>1.0270976  </td></tr>
	<tr><th scope=row>day</th><td>4          </td><td>100        </td><td>50.500000  </td><td>29.011492  </td><td>50.50      </td><td>50.500000  </td><td>37.065000  </td><td> 1.00      </td><td>100.00     </td><td>99.00      </td><td> 0.00000000</td><td>-1.2360552 </td><td>2.9011492  </td></tr>
</tbody>
</table>



by default, in describe, na.rm = TRUE


```R
cor(parenthood2)
```


<table>
<thead><tr><th></th><th scope=col>dan.sleep</th><th scope=col>baby.sleep</th><th scope=col>dan.grump</th><th scope=col>day</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td> 1</td><td>NA</td><td>NA</td><td>NA</td></tr>
	<tr><th scope=row>baby.sleep</th><td>NA</td><td> 1</td><td>NA</td><td>NA</td></tr>
	<tr><th scope=row>dan.grump</th><td>NA</td><td>NA</td><td> 1</td><td>NA</td></tr>
	<tr><th scope=row>day</th><td>NA</td><td>NA</td><td>NA</td><td> 1</td></tr>
</tbody>
</table>



to handle NAs here, specify use argument

use = "complete.obs" -> R ignores all cases that have ANY missing values (i.e. competely remove rows that have an NA, then do the correlation)<br>
use = "pairwise.complete.obs" -> R will ignore NAs observation-by-observation


```R
cor(parenthood2, use="complete.obs")
```


<table>
<thead><tr><th></th><th scope=col>dan.sleep</th><th scope=col>baby.sleep</th><th scope=col>dan.grump</th><th scope=col>day</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td> 1.00000000</td><td> 0.6394985 </td><td>-0.89951468</td><td> 0.06132891</td></tr>
	<tr><th scope=row>baby.sleep</th><td> 0.63949845</td><td> 1.0000000 </td><td>-0.58656066</td><td> 0.14555814</td></tr>
	<tr><th scope=row>dan.grump</th><td>-0.89951468</td><td>-0.5865607 </td><td> 1.00000000</td><td>-0.06816586</td></tr>
	<tr><th scope=row>day</th><td> 0.06132891</td><td> 0.1455581 </td><td>-0.06816586</td><td> 1.00000000</td></tr>
</tbody>
</table>




```R
cor(parenthood2, use="pairwise.complete.obs")
```


<table>
<thead><tr><th></th><th scope=col>dan.sleep</th><th scope=col>baby.sleep</th><th scope=col>dan.grump</th><th scope=col>day</th></tr></thead>
<tbody>
	<tr><th scope=row>dan.sleep</th><td> 1.00000000 </td><td> 0.61472303 </td><td>-0.903442442</td><td>-0.076796665</td></tr>
	<tr><th scope=row>baby.sleep</th><td> 0.61472303 </td><td> 1.00000000 </td><td>-0.567802669</td><td> 0.058309485</td></tr>
	<tr><th scope=row>dan.grump</th><td>-0.90344244 </td><td>-0.56780267 </td><td> 1.000000000</td><td> 0.005833399</td></tr>
	<tr><th scope=row>day</th><td>-0.07679667 </td><td> 0.05830949 </td><td> 0.005833399</td><td> 1.000000000</td></tr>
</tbody>
</table>



correlate() uses pairwise.complete by default

pairwise.complete keeps more observations, but for correlation matrices, every correlation is derived from a slightly different subset of the data

which method to use depends on WHY you suspect the data is missing
