# Chapter 3

## Vectors


```R
sales.by.month <- c(0, 100, 200, 50, 0, 0, 0, 0, 0, 0, 0, 0)
sales.by.month
```


<ol class=list-inline>
	<li>0</li>
	<li>100</li>
	<li>200</li>
	<li>50</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
</ol>



Indexing - just like python


```R
febuary.sales <- sales.by.month[2]
febuary.sales
```


100


altering elements of a vector


```R
sales.by.month[5] <- 25
sales.by.month
```


<ol class=list-inline>
	<li>0</li>
	<li>100</li>
	<li>200</li>
	<li>50</li>
	<li>25</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
</ol>



does this automatically do a deep copy?


```R
test_var <- sales.by.month[10]
print(test_var)

sales.by.month[10] <- 25
print(test_var)
```

    [1] 0
    [1] 0


Other useful stuff about vectors


```R
# length
len = length(sales.by.month)
print(len)

# alter all elements at once
sales_mult = sales.by.month * 7
print(sales_mult)

# dividing two vectors
days.per.month <- c(31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
profit <- sales.by.month * 7
daily = profit / days.per.month
print(daily)
```

    [1] 12
     [1]    0  700 1400  350  175    0    0    0    0  175    0    0
     [1]  0.000000 25.000000 45.161290 11.666667  5.645161  0.000000  0.000000
     [8]  0.000000  0.000000  5.645161  0.000000  0.000000


When initially writing this, I missed a zero in sales.by.month. R only threw a warning, not an exception when the vectors were different lengths

## Text data

Strings are single elements


```R
greeting <- "hello"
print(greeting[1])
```

    [1] "hello"


vector of strings


```R
months <- c("January", "Feburary", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
print(months)
```

     [1] "January"   "Feburary"  "March"     "April"     "May"       "June"     
     [7] "July"      "August"    "September" "October"   "November"  "December" 



```R
print(months[4])
```

    [1] "April"


number of chars in a string


```R
nchar(greeting)
```


5


nchar on vectors


```R
nchar(months)
```


<ol class=list-inline>
	<li>7</li>
	<li>8</li>
	<li>5</li>
	<li>5</li>
	<li>3</li>
	<li>4</li>
	<li>4</li>
	<li>6</li>
	<li>9</li>
	<li>7</li>
	<li>8</li>
	<li>8</li>
</ol>



## Boolean data

basically the same as python, except is doesnt look like you can use `and` or `or`.<br>
reserved words for booleans are TRUE or FALSE (all caps)<br>
can do T or F as a shortcut (but they are not reserved words)

vectors of booleans (again, basically the same a python


```R
x <- c(TRUE, TRUE, TRUE)
x
```


<ol class=list-inline>
	<li>TRUE</li>
	<li>TRUE</li>
	<li>TRUE</li>
</ol>




```R
sales.by.month > 0
```


<ol class=list-inline>
	<li>FALSE</li>
	<li>TRUE</li>
	<li>TRUE</li>
	<li>TRUE</li>
	<li>TRUE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>TRUE</li>
	<li>FALSE</li>
	<li>FALSE</li>
</ol>



## Indexing


```R
sales.by.month[c(4, 3, 2)]
```


<ol class=list-inline>
	<li>50</li>
	<li>200</li>
	<li>100</li>
</ol>




```R
sales.by.month[2:8]
```


<ol class=list-inline>
	<li>100</li>
	<li>200</li>
	<li>50</li>
	<li>25</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
</ol>



indexing seems to be the same as with numpy


can also do logical indexing


```R
months[sales.by.month > 0]
```


<ol class=list-inline>
	<li>'Feburary'</li>
	<li>'March'</li>
	<li>'April'</li>
	<li>'May'</li>
	<li>'October'</li>
</ol>




```R
stock.levels = c("high", "high", "low", "out", "out", "high", "high", "high", "high", "high", "high", "high")

months[stock.levels == "out"]
```


<ol class=list-inline>
	<li>'April'</li>
	<li>'May'</li>
</ol>




```R
months[stock.levels == "out" | stock.levels == "low"]
```


<ol class=list-inline>
	<li>'March'</li>
	<li>'April'</li>
	<li>'May'</li>
</ol>



# Chapter 4

\# = comment


```R
seeker <- 3.14158        # create the first variable
lover <- 2.7183          # create the second variable
keeper <- seeker * lover # now multiply them to create a third one
print(keeper)
```

    [1] 8.539757


## Installing / loading packages

(slightly easier with R studio)


```R
exists("read.spss")
```


FALSE



```R
library(foreign)
```


```R
exists("read.spss")
```


TRUE



```R
detach("package:foreign", unload=T)
exists("read.spss")
```


FALSE


Downloading packages


```R
# install.packages("psych")
```


```R
# install.packages("lsr")
# install.packages("car")
```

## Managing the workspace


```R
objects()
```


<ol class=list-inline>
	<li>'daily'</li>
	<li>'days.per.month'</li>
	<li>'febuary.sales'</li>
	<li>'greeting'</li>
	<li>'keeper'</li>
	<li>'len'</li>
	<li>'lover'</li>
	<li>'months'</li>
	<li>'profit'</li>
	<li>'sales_mult'</li>
	<li>'sales.by.month'</li>
	<li>'seeker'</li>
	<li>'stock.levels'</li>
	<li>'test_var'</li>
	<li>'x'</li>
</ol>




```R
library(lsr)
who()
```


       -- Name --       -- Class --   -- Size --
       daily            numeric       12        
       days.per.month   numeric       12        
       febuary.sales    numeric        1        
       greeting         character      1        
       keeper           numeric        1        
       len              integer        1        
       lover            numeric        1        
       months           character     12        
       profit           numeric       12        
       sales_mult       numeric       12        
       sales.by.month   numeric       12        
       seeker           numeric        1        
       stock.levels     character     12        
       test_var         numeric        1        
       x                logical        3        


removing variables from the workspace


```R
rm(seeker, lover)
```


```R
who()
```


       -- Name --       -- Class --   -- Size --
       daily            numeric       12        
       days.per.month   numeric       12        
       febuary.sales    numeric        1        
       greeting         character      1        
       keeper           numeric        1        
       len              integer        1        
       months           character     12        
       profit           numeric       12        
       sales_mult       numeric       12        
       sales.by.month   numeric       12        
       stock.levels     character     12        
       test_var         numeric        1        
       x                logical        3        


## Navigating the file system


```R
getwd()
```


'/home/alex/Documents/GitHub/Data-science-notes/R'



```R
setwd("..")
```


```R
getwd()
```


'/home/alex/Documents/GitHub/Data-science-notes'



```R
setwd("./R")
```


```R
getwd()
```


'/home/alex/Documents/GitHub/Data-science-notes/R'



```R
list.files()
```


<ol class=list-inline>
	<li>'booksales.Rdata'</li>
	<li>'Learning Statistics with R chapter 3-4.ipynb'</li>
</ol>




```R
setwd("..")
list.files()
```


<ol class=list-inline>
	<li>'Other'</li>
	<li>'Python'</li>
	<li>'R'</li>
	<li>'Statistics'</li>
	<li>'Tableau'</li>
	<li>'textbooks'</li>
</ol>




```R
path.expand("~")
```


'/home/alex'


## Loading and saving data

3 important file types:<br>
- workspace files (.Rdata) - saves whole workspace (data and variables)
- .csv
- script files: .R


```R
getwd()
```


'/home/alex/Documents/GitHub/Data-science-notes'



```R
setwd("./R")
```


```R
getwd()
```


'/home/alex/Documents/GitHub/Data-science-notes/R'



```R
objects()
```


<ol class=list-inline>
	<li>'daily'</li>
	<li>'days.per.month'</li>
	<li>'febuary.sales'</li>
	<li>'greeting'</li>
	<li>'keeper'</li>
	<li>'len'</li>
	<li>'months'</li>
	<li>'profit'</li>
	<li>'sales_mult'</li>
	<li>'sales.by.month'</li>
	<li>'stock.levels'</li>
	<li>'test_var'</li>
	<li>'x'</li>
</ol>



Use load to load a workspace file


```R
load("booksales.Rdata")
```


```R
who()
```


       -- Name --               -- Class --   -- Size --
       any.sales.this.month     logical       12        
       average.daily.sales      numeric       12        
       daily                    numeric       12        
       days.per.month           numeric       12        
       february.sales           numeric        1        
       febuary.sales            numeric        1        
       greeting                 character      1        
       hourly.wage              numeric        1        
       hours.writing.per.week   numeric        1        
       keeper                   numeric        1        
       len                      integer        1        
       months                   character     12        
       profit                   numeric       12        
       revenue                  numeric        1        
       royalty                  numeric        1        
       sales                    numeric        1        
       sales_mult               numeric       12        
       sales.by.month           numeric       12        
       stock.levels             character     12        
       test_var                 numeric        1        
       total.writing.hours      numeric        1        
       weeks.writing            numeric        1        
       x                        logical        3        


Loading a csv file

use read.csv(),read.table() does basically the same thing


```R
books <- read.csv(file="booksales.csv")
books
```


<table>
<thead><tr><th scope=col>Month</th><th scope=col>Days</th><th scope=col>Sales</th><th scope=col>Stock.Levels</th></tr></thead>
<tbody>
	<tr><td>January  </td><td>31       </td><td>  0      </td><td>high     </td></tr>
	<tr><td>February </td><td>28       </td><td>100      </td><td>high     </td></tr>
	<tr><td>March    </td><td>31       </td><td>200      </td><td>low      </td></tr>
	<tr><td>April    </td><td>30       </td><td> 50      </td><td>out      </td></tr>
	<tr><td>May      </td><td>31       </td><td>  0      </td><td>out      </td></tr>
	<tr><td>June     </td><td>30       </td><td>  0      </td><td>high     </td></tr>
	<tr><td>July     </td><td>31       </td><td>  0      </td><td>high     </td></tr>
	<tr><td>August   </td><td>31       </td><td>  0      </td><td>high     </td></tr>
	<tr><td>September</td><td>30       </td><td>  0      </td><td>high     </td></tr>
	<tr><td>October  </td><td>31       </td><td>  0      </td><td>high     </td></tr>
	<tr><td>November </td><td>30       </td><td>  0      </td><td>high     </td></tr>
	<tr><td>December </td><td>31       </td><td>  0      </td><td>high     </td></tr>
</tbody>
</table>



books is a DataFrame- this is what inspired pandas

Saving workspace files


```R
who()
```


       -- Name --               -- Class --   -- Size --
       any.sales.this.month     logical       12        
       average.daily.sales      numeric       12        
       books                    data.frame    12 x 4    
       daily                    numeric       12        
       days.per.month           numeric       12        
       february.sales           numeric       1         
       febuary.sales            numeric       1         
       greeting                 character     1         
       hourly.wage              numeric       1         
       hours.writing.per.week   numeric       1         
       keeper                   numeric       1         
       len                      integer       1         
       months                   character     12        
       profit                   numeric       12        
       revenue                  numeric       1         
       royalty                  numeric       1         
       sales                    numeric       1         
       sales_mult               numeric       12        
       sales.by.month           numeric       12        
       stock.levels             character     12        
       test_var                 numeric       1         
       total.writing.hours      numeric       1         
       weeks.writing            numeric       1         
       x                        logical       3         



```R
save.image("all_vars_20210713.Rdata")
```

save some variables


```R
save(sales_mult, weeks.writing, file="somedata_20210713.Rdata")
```

## Other useful things about variables

special values: <br>
- `Inf`
- `NaN` (not a number- still treated as numeric) - there is a value but its insane (0/0)
- `NA`(not available- something is missing) - there is supposed to be a value there
- `NULL` - No value whatsoever

naming vector elements


```R
profit <- c(3.1, 0.1, -1.4, 1.1)
print(profit)
```

    [1]  3.1  0.1 -1.4  1.1



```R
names(profit) <- c("Q1", "Q2", "Q3", "Q4")
print(profit)
```

      Q1   Q2   Q3   Q4 
     3.1  0.1 -1.4  1.1 


This is kind of like the indices for a pandas series

delete the names


```R
names(profit) <- NULL
print(profit)
```

    [1]  3.1  0.1 -1.4  1.1



```R
profit <- c("Q1" = 3.1, "Q2" = 0.1, "Q3" = -1.4, "Q4" = 1.1)
print(profit)
```

      Q1   Q2   Q3   Q4 
     3.1  0.1 -1.4  1.1 



```R
profit["Q1"]
```


<strong>Q1:</strong> 3.1


These names work a lot like key-value pairs in a python dictionary

variable classes:

- class()
- mode()
- type()


```R
class(profit["Q1"])
```


'numeric'



```R
mode(profit["Q1"])
```


'numeric'



```R
typeof(profit["Q1"])
```


'double'


## Factors

numeric already works with ratio scale data


```R
# response time for 5 different events
RT <- c(342, 401, 590, 391, 554)
```

May also work with interval scale data- just remember not to do any multiplication or division since that would be meaningless

can also tolerate numerical values for ordinal data (i.e. ranking items)

absolutely terrible for nominal scale data- that's where **factors** come in

ex: tracking who had what set of instructions


```R
group <- c(1, 1, 1, 2, 2, 2, 3, 3, 3)
group <- as.factor(group)
group
```


<ol class=list-inline>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
</ol>

<details>
	<summary style=display:list-item;cursor:pointer>
		<strong>Levels</strong>:
	</summary>
	<ol class=list-inline>
		<li>'1'</li>
		<li>'2'</li>
		<li>'3'</li>
	</ol>
</details>



```R
class(group)
```


'factor'



```R
group + 2
```

    Warning message in Ops.factor(group, 2):
    “‘+’ not meaningful for factors”


<ol class=list-inline>
	<li>&lt;NA&gt;</li>
	<li>&lt;NA&gt;</li>
	<li>&lt;NA&gt;</li>
	<li>&lt;NA&gt;</li>
	<li>&lt;NA&gt;</li>
	<li>&lt;NA&gt;</li>
	<li>&lt;NA&gt;</li>
	<li>&lt;NA&gt;</li>
	<li>&lt;NA&gt;</li>
</ol>



**levels**


```R
gender <- as.factor(c(1, 1, 1, 1, 1, 2, 2, 2, 2))
print(gender)
```

    [1] 1 1 1 1 1 2 2 2 2
    Levels: 1 2



```R
levels(group) <- c("group 1", "group 2", "group 3")
print(group)
```

    [1] group 1 group 1 group 1 group 2 group 2 group 2 group 3 group 3 group 3
    Levels: group 1 group 2 group 3



```R
levels(gender) <- c("male", "female")
print(gender)
```

    [1] male   male   male   male   male   female female female female
    Levels: male female


## Data frames


```R
age <- c(17, 19, 21, 37, 18, 19, 47, 18, 19)
score <- c(12, 10, 11, 15, 16, 14, 25, 21, 29)
```


```R
expt <- data.frame(age, gender, group, score)
expt
```


<table>
<thead><tr><th scope=col>age</th><th scope=col>gender</th><th scope=col>group</th><th scope=col>score</th></tr></thead>
<tbody>
	<tr><td>17     </td><td>male   </td><td>group 1</td><td>12     </td></tr>
	<tr><td>19     </td><td>male   </td><td>group 1</td><td>10     </td></tr>
	<tr><td>21     </td><td>male   </td><td>group 1</td><td>11     </td></tr>
	<tr><td>37     </td><td>male   </td><td>group 2</td><td>15     </td></tr>
	<tr><td>18     </td><td>male   </td><td>group 2</td><td>16     </td></tr>
	<tr><td>19     </td><td>female </td><td>group 2</td><td>14     </td></tr>
	<tr><td>47     </td><td>female </td><td>group 3</td><td>25     </td></tr>
	<tr><td>18     </td><td>female </td><td>group 3</td><td>21     </td></tr>
	<tr><td>19     </td><td>female </td><td>group 3</td><td>29     </td></tr>
</tbody>
</table>



expt is completely independent from the variables used to make it. Changing or deleting `age` will NOT change the age column in expt

Extract column from expt


```R
expt$score
```


<ol class=list-inline>
	<li>12</li>
	<li>10</li>
	<li>11</li>
	<li>15</li>
	<li>16</li>
	<li>14</li>
	<li>25</li>
	<li>21</li>
	<li>29</li>
</ol>




```R
names(expt)
```


<ol class=list-inline>
	<li>'age'</li>
	<li>'gender'</li>
	<li>'group'</li>
	<li>'score'</li>
</ol>




```R
who(expand = TRUE)
```


       -- Name --               -- Class --   -- Size --
       age                      numeric       9         
       any.sales.this.month     logical       12        
       average.daily.sales      numeric       12        
       books                    data.frame    12 x 4    
        $Month                  factor        12        
        $Days                   integer       12        
        $Sales                  integer       12        
        $Stock.Levels           factor        12        
       daily                    numeric       12        
       days.per.month           numeric       12        
       expt                     data.frame    9 x 4     
        $age                    numeric       9         
        $gender                 factor        9         
        $group                  factor        9         
        $score                  numeric       9         
       february.sales           numeric       1         
       febuary.sales            numeric       1         
       gender                   factor        9         
       greeting                 character     1         
       group                    factor        9         
       hourly.wage              numeric       1         
       hours.writing.per.week   numeric       1         
       keeper                   numeric       1         
       len                      integer       1         
       months                   character     12        
       profit                   numeric       4         
       revenue                  numeric       1         
       royalty                  numeric       1         
       RT                       numeric       5         
       sales                    numeric       1         
       sales_mult               numeric       12        
       sales.by.month           numeric       12        
       score                    numeric       9         
       stock.levels             character     12        
       test_var                 numeric       1         
       total.writing.hours      numeric       1         
       weeks.writing            numeric       1         
       x                        logical       3         


## Lists

access elements the same way you would access them in a df


```R
Danielle <- list(age=34, nerd=TRUE, parents=c("Joe", "Liz"))
print(Danielle)
```

    $age
    [1] 34
    
    $nerd
    [1] TRUE
    
    $parents
    [1] "Joe" "Liz"
    



```R
Danielle$nerd
```


TRUE


new entry into list


```R
Danielle$children <- "Bob"
```

Lists can also contain other lists

## Formulas

a variable that specifies a relationship between 2 other variables.


```R
formula1 <- out ~ pred
formula1
```


    out ~ pred


R doesn't care if out or pred actually exist

Generally, this means "The out (outcome) variable, analyzed in terms of the pred (predictor) variable"

other formats:


```R
formula2 <- out ~ pred1 + pred2
formula3 <- out ~ pred1 * pred2
formula4 <- ~ var1 + var2
```

## Generic functions

print(), summary(), plot(), ect. 

Changes behavior based on class


```R
my.formula <- blah ~ blah.blah
print(my.formula)
```

    blah ~ blah.blah


print() checks the class of the var, then calls the method associated with that class (method dispatch)


```R
print.formula(my.formula)
```


    Error in print.formula(my.formula): could not find function "print.formula"
    Traceback:



(I'm assuming this is some kind of version issue)

If no dedicated method could be found, then resort to the default function


```R
print.default(my.formula)
```

    blah ~ blah.blah
    attr(,"class")
    [1] "formula"
    attr(,".Environment")
    <environment: R_GlobalEnv>


## help documentation


```R
?load
```


```R
help(load)
```

fuzzy search


```R
??load
```


```R
help.search("load")
```
