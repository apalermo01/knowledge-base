# Chapter 7: pragmatic matters

## Tabulating and cross-tabulating data

**Creating tables from vectors**


```R
library(lsr)
load("nightgarden.Rdata")
who()
```


       -- Name --   -- Class --   -- Size --
       speaker      character     10        
       utterance    character     10        



```R
print(speaker)
```

     [1] "upsy-daisy"  "upsy-daisy"  "upsy-daisy"  "upsy-daisy"  "tombliboo"  
     [6] "tombliboo"   "makka-pakka" "makka-pakka" "makka-pakka" "makka-pakka"



```R
print(utterance)
```

     [1] "pip" "pip" "onk" "onk" "ee"  "oo"  "pip" "pip" "onk" "onk"



```R
table(speaker)
```


    speaker
    makka-pakka   tombliboo  upsy-daisy 
              4           2           4 



```R
table(speaker, utterance)
```


                 utterance
    speaker       ee onk oo pip
      makka-pakka  0   2  0   2
      tombliboo    1   0  1   0
      upsy-daisy   0   2  0   2


**Creating tables from data frames**


```R
itng <- data.frame(speaker, utterance)
itng
```


<table>
<thead><tr><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><td>upsy-daisy </td><td>pip        </td></tr>
	<tr><td>upsy-daisy </td><td>pip        </td></tr>
	<tr><td>upsy-daisy </td><td>onk        </td></tr>
	<tr><td>upsy-daisy </td><td>onk        </td></tr>
	<tr><td>tombliboo  </td><td>ee         </td></tr>
	<tr><td>tombliboo  </td><td>oo         </td></tr>
	<tr><td>makka-pakka</td><td>pip        </td></tr>
	<tr><td>makka-pakka</td><td>pip        </td></tr>
	<tr><td>makka-pakka</td><td>onk        </td></tr>
	<tr><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>




```R
table(itng)
```


                 utterance
    speaker       ee onk oo pip
      makka-pakka  0   2  0   2
      tombliboo    1   0  1   0
      upsy-daisy   0   2  0   2


tabulate specific variables


```R
xtabs(formula=~speaker+utterance, data=itng)
```


                 utterance
    speaker       ee onk oo pip
      makka-pakka  0   2  0   2
      tombliboo    1   0  1   0
      upsy-daisy   0   2  0   2


**Converting a table of counts to a table of porportions**


```R
itng.table <- table(itng)
itng.table
```


                 utterance
    speaker       ee onk oo pip
      makka-pakka  0   2  0   2
      tombliboo    1   0  1   0
      upsy-daisy   0   2  0   2



```R
prop.table(itng.table)
```


                 utterance
    speaker        ee onk  oo pip
      makka-pakka 0.0 0.2 0.0 0.2
      tombliboo   0.1 0.0 0.1 0.0
      upsy-daisy  0.0 0.2 0.0 0.2


porortion by row


```R
prop.table(itng.table, margin=1)
```


                 utterance
    speaker        ee onk  oo pip
      makka-pakka 0.0 0.5 0.0 0.5
      tombliboo   0.5 0.0 0.5 0.0
      upsy-daisy  0.0 0.5 0.0 0.5


porportion by column


```R
prop.table(itng.table, margin=2)
```


                 utterance
    speaker        ee onk  oo pip
      makka-pakka 0.0 0.5 0.0 0.5
      tombliboo   1.0 0.0 1.0 0.0
      upsy-daisy  0.0 0.5 0.0 0.5


## Transforming and recoding a variable


```R
load("likert.Rdata")
likert.raw
```


<ol class=list-inline>
	<li>1</li>
	<li>7</li>
	<li>3</li>
	<li>4</li>
	<li>4</li>
	<li>4</li>
	<li>2</li>
	<li>6</li>
	<li>5</li>
	<li>5</li>
</ol>



center the likert data on 4 since it's "no opinion"


```R
likert.centered <- likert.raw-4
likert.centered
```


<ol class=list-inline>
	<li>-3</li>
	<li>3</li>
	<li>-1</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>-2</li>
	<li>2</li>
	<li>1</li>
	<li>1</li>
</ol>



strength of opinion


```R
opinion.strength <- abs(likert.centered)
opinion.strength
```


<ol class=list-inline>
	<li>3</li>
	<li>3</li>
	<li>1</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>1</li>
	<li>1</li>
</ol>



direction of opinion, ignore strength


```R
opinion.dir <- sign(likert.centered)
opinion.dir
```


<ol class=list-inline>
	<li>-1</li>
	<li>1</li>
	<li>-1</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>-1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
</ol>



**Cutting a numeric variable into categories**


```R
age <- c(60, 58, 24, 26, 34, 42, 31, 30, 33, 2, 9)

age.breaks <- seq(from=0, to=60, by=20)
age.breaks
```


<ol class=list-inline>
	<li>0</li>
	<li>20</li>
	<li>40</li>
	<li>60</li>
</ol>




```R
age.labels <- c("young", "adult", "older")
age.labels
```


<ol class=list-inline>
	<li>'young'</li>
	<li>'adult'</li>
	<li>'older'</li>
</ol>




```R
age.group <- cut(x=age, breaks=age.breaks, labels=age.labels)
data.frame(age, age.group)

```


<table>
<thead><tr><th scope=col>age</th><th scope=col>age.group</th></tr></thead>
<tbody>
	<tr><td>60   </td><td>older</td></tr>
	<tr><td>58   </td><td>older</td></tr>
	<tr><td>24   </td><td>adult</td></tr>
	<tr><td>26   </td><td>adult</td></tr>
	<tr><td>34   </td><td>adult</td></tr>
	<tr><td>42   </td><td>older</td></tr>
	<tr><td>31   </td><td>adult</td></tr>
	<tr><td>30   </td><td>adult</td></tr>
	<tr><td>33   </td><td>adult</td></tr>
	<tr><td> 2   </td><td>young</td></tr>
	<tr><td> 9   </td><td>young</td></tr>
</tbody>
</table>




```R
table(age.group)
```


    age.group
    young adult older 
        2     6     3 


R can do that for us


```R
age.group2 <- cut(age, breaks=3)
table(age.group2)
```


    age.group2
    (1.94,21.3] (21.3,40.7] (40.7,60.1] 
              2           6           3 


separate into roughly equal numbers of people

use quantileCut() in lsr package


```R
age.group3 <- quantileCut(age, n=3)
table(age.group3)
```


    age.group3
    (1.94,27.3] (27.3,33.7] (33.7,60.1] 
              4           3           4 


## A few more mathematical functions and operations

- sqrt() - square root
- abs() - absolute value
- log10() - log base 10
- log() - log (base=3 by default
- exp() - exponentiation
- round() - round to nearest; use digits to specify number of digits to round to
- signif() - round to selected number of significant digits
- floor() - round down
- ceiling() - round up

- %/% - integer division
- % - modulus

## Extacting a subset of a vector


```R
is.MP.speaking <- speaker == 'makka-pakka'
is.MP.speaking
```


<ol class=list-inline>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>FALSE</li>
	<li>TRUE</li>
	<li>TRUE</li>
	<li>TRUE</li>
	<li>TRUE</li>
</ol>




```R
utterance[is.MP.speaking]
```


<ol class=list-inline>
	<li>'pip'</li>
	<li>'pip'</li>
	<li>'onk'</li>
	<li>'onk'</li>
</ol>




```R
utterance[speaker == 'makka-pakka']
```


<ol class=list-inline>
	<li>'pip'</li>
	<li>'pip'</li>
	<li>'onk'</li>
	<li>'onk'</li>
</ol>



**%in% operator**

similar to == but can match multiple values


```R
speaker[utterance %in% c("pip", "oo")]
```


<ol class=list-inline>
	<li>'upsy-daisy'</li>
	<li>'upsy-daisy'</li>
	<li>'tombliboo'</li>
	<li>'makka-pakka'</li>
	<li>'makka-pakka'</li>
</ol>



pick elements 2 and 3


```R
utterance[2:3]
```


<ol class=list-inline>
	<li>'pip'</li>
	<li>'onk'</li>
</ol>



drop elements 2 and 3


```R
utterance[-(2:3)]
```


<ol class=list-inline>
	<li>'pip'</li>
	<li>'onk'</li>
	<li>'ee'</li>
	<li>'oo'</li>
	<li>'pip'</li>
	<li>'pip'</li>
	<li>'onk'</li>
	<li>'onk'</li>
</ol>



**Splitting a vector by group**

split(x=variable that nees to be plit into groups, f=grouping variable)


```R
speech.by.char <- split(x=utterance, f=speaker)
print(speech.by.char)
```

    $`makka-pakka`
    [1] "pip" "pip" "onk" "onk"
    
    $tombliboo
    [1] "ee" "oo"
    
    $`upsy-daisy`
    [1] "pip" "pip" "onk" "onk"
    


first utterance by makka-pakka:


```R
speech.by.char$'makka-pakka'[1]
```


'pip'



```R
speech.by.char$tombliboo
```


<ol class=list-inline>
	<li>'ee'</li>
	<li>'oo'</li>
</ol>



note: R requires the quotes when the original record had a space

use importList() from the lsr package to import these split variables into the workspace


```R
who()
```


       -- Name --         -- Class --   -- Size --
       age                numeric       11        
       age.breaks         numeric       4         
       age.group          factor        11        
       age.group2         factor        11        
       age.group3         factor        11        
       age.labels         character     3         
       is.MP.speaking     logical       10        
       itng               data.frame    10 x 2    
       itng.table         table         3 x 4     
       likert.centered    numeric       10        
       likert.raw         numeric       10        
       opinion.dir        numeric       10        
       opinion.strength   numeric       10        
       speaker            character     10        
       speech.by.char     list          3         
       utterance          character     10        



```R
importList(speech.by.char)
```

    Create these variables? [y/n] 
    Create these variables? [y/n] 
    Create these variables? [y/n] 
    Create these variables? [y/n] y
    Names of variables to be created:
    [1] "makka.pakka" "tombliboo"   "upsy.daisy" 



```R
who()
```


       -- Name --         -- Class --   -- Size --
       age                numeric       11        
       age.breaks         numeric       4         
       age.group          factor        11        
       age.group2         factor        11        
       age.group3         factor        11        
       age.labels         character     3         
       is.MP.speaking     logical       10        
       itng               data.frame    10 x 2    
       itng.table         table         3 x 4     
       likert.centered    numeric       10        
       likert.raw         numeric       10        
       makka.pakka        character     4         
       opinion.dir        numeric       10        
       opinion.strength   numeric       10        
       speaker            character     10        
       speech.by.char     list          3         
       tombliboo          character     2         
       upsy.daisy         character     4         
       utterance          character     10        



```R
makka.pakka
```


<ol class=list-inline>
	<li>'pip'</li>
	<li>'pip'</li>
	<li>'onk'</li>
	<li>'onk'</li>
</ol>



## Extracting a subset of a data frame

note: this is pretty much identical to indexing with pandas

**subset function**

x = data frame<br> 
subset = vector of logical values indicating cases (i.e. rows) to keep<br> 
select = indicates which varialbes (columns) to keep



```R
df <- subset(x=itng,
            subset= speaker=='makka-pakka',
            select=utterance)
```


```R
print(df)
```

       utterance
    7        pip
    8        pip
    9        onk
    10       onk


note that row numbers are preserved


```R
subset(x=itng,
      subset=speaker=='makka-pakka')
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><th scope=row>7</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>8</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>9</th><td>makka-pakka</td><td>onk        </td></tr>
	<tr><th scope=row>10</th><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>




```R
subset(x=itng, select=utterance)
```


<table>
<thead><tr><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><td>pip</td></tr>
	<tr><td>pip</td></tr>
	<tr><td>onk</td></tr>
	<tr><td>onk</td></tr>
	<tr><td>ee </td></tr>
	<tr><td>oo </td></tr>
	<tr><td>pip</td></tr>
	<tr><td>pip</td></tr>
	<tr><td>onk</td></tr>
	<tr><td>onk</td></tr>
</tbody>
</table>



**using square brackets 1. rows and columns**


```R
load("nightgarden2.Rdata")
```


```R
garden
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th><th scope=col>line</th></tr></thead>
<tbody>
	<tr><th scope=row>case.1</th><td>upsy-daisy </td><td>pip        </td><td>1          </td></tr>
	<tr><th scope=row>case.2</th><td>upsy-daisy </td><td>pip        </td><td>2          </td></tr>
	<tr><th scope=row>case.3</th><td>tombliboo  </td><td>ee         </td><td>5          </td></tr>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td><td>7          </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td><td>9          </td></tr>
</tbody>
</table>




```R
garden[4:5, 1:2]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>




```R
garden[c(4,5), c(1,2)]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>




```R
garden[c("case.4", "case.5"), c("speaker", "utterance")]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>




```R
garden[4:5, c("speaker", "utterance")]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>




```R
is.MP.speaking <- garden$speaker == "makka-pakka"
garden[is.MP.speaking, c("speaker", "utterance")]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>



**Using square brackets 2: some elaborations**


```R
garden[,1:2]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><th scope=row>case.1</th><td>upsy-daisy </td><td>pip        </td></tr>
	<tr><th scope=row>case.2</th><td>upsy-daisy </td><td>pip        </td></tr>
	<tr><th scope=row>case.3</th><td>tombliboo  </td><td>ee         </td></tr>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>




```R
garden[4:5,]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th><th scope=col>line</th></tr></thead>
<tbody>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td><td>7          </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td><td>9          </td></tr>
</tbody>
</table>



delete 3rd column


```R
garden[,-3]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><th scope=row>case.1</th><td>upsy-daisy </td><td>pip        </td></tr>
	<tr><th scope=row>case.2</th><td>upsy-daisy </td><td>pip        </td></tr>
	<tr><th scope=row>case.3</th><td>tombliboo  </td><td>ee         </td></tr>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>



**Using square brackets: 3. understanding 'dropping'**


```R
garden[5,]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th><th scope=col>line</th></tr></thead>
<tbody>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td><td>9          </td></tr>
</tbody>
</table>




```R
garden
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th><th scope=col>line</th></tr></thead>
<tbody>
	<tr><th scope=row>case.1</th><td>upsy-daisy </td><td>pip        </td><td>1          </td></tr>
	<tr><th scope=row>case.2</th><td>upsy-daisy </td><td>pip        </td><td>2          </td></tr>
	<tr><th scope=row>case.3</th><td>tombliboo  </td><td>ee         </td><td>5          </td></tr>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td><td>7          </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td><td>9          </td></tr>
</tbody>
</table>




```R
garden[,3]
```


<ol class=list-inline>
	<li>1</li>
	<li>2</li>
	<li>5</li>
	<li>7</li>
	<li>9</li>
</ol>



R noticed that the outpute doesn't need a data frame becuase it's only one variable. 


```R
garden[,3,drop=FALSE]
```


<table>
<thead><tr><th></th><th scope=col>line</th></tr></thead>
<tbody>
	<tr><th scope=row>case.1</th><td>1</td></tr>
	<tr><th scope=row>case.2</th><td>2</td></tr>
	<tr><th scope=row>case.3</th><td>5</td></tr>
	<tr><th scope=row>case.4</th><td>7</td></tr>
	<tr><th scope=row>case.5</th><td>9</td></tr>
</tbody>
</table>



**Using square brackets: 4. columns only**


```R
garden[1:2]
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th></tr></thead>
<tbody>
	<tr><th scope=row>case.1</th><td>upsy-daisy </td><td>pip        </td></tr>
	<tr><th scope=row>case.2</th><td>upsy-daisy </td><td>pip        </td></tr>
	<tr><th scope=row>case.3</th><td>tombliboo  </td><td>ee         </td></tr>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td></tr>
</tbody>
</table>




```R
garden[3]
```


<table>
<thead><tr><th></th><th scope=col>line</th></tr></thead>
<tbody>
	<tr><th scope=row>case.1</th><td>1</td></tr>
	<tr><th scope=row>case.2</th><td>2</td></tr>
	<tr><th scope=row>case.3</th><td>5</td></tr>
	<tr><th scope=row>case.4</th><td>7</td></tr>
	<tr><th scope=row>case.5</th><td>9</td></tr>
</tbody>
</table>




```R
garden[[3]]
```


<ol class=list-inline>
	<li>1</li>
	<li>2</li>
	<li>5</li>
	<li>7</li>
	<li>9</li>
</ol>



## Sorting, flipping, and merging data

**Sorting a numeric or character vector**


```R
numbers <- c(2, 4, 3)
sort(numbers)
```


<ol class=list-inline>
	<li>2</li>
	<li>3</li>
	<li>4</li>
</ol>




```R
sort(numbers, decreasing=TRUE)
```


<ol class=list-inline>
	<li>4</li>
	<li>3</li>
	<li>2</li>
</ol>




```R
text <- c("aardvark", "zebra", "swing")
sort(text)
```


<ol class=list-inline>
	<li>'aardvark'</li>
	<li>'swing'</li>
	<li>'zebra'</li>
</ol>



**Sorting a factor**


```R
fac <- factor(text)
print(fac)
```

    [1] aardvark zebra    swing   
    Levels: aardvark swing zebra



```R
print(sort(fac))
```

    [1] aardvark swing    zebra   
    Levels: aardvark swing zebra



```R
fac <- factor(text, levels=c("zebra", "swing", "aardvark"))
print(fac)
```

    [1] aardvark zebra    swing   
    Levels: zebra swing aardvark



```R
print(sort(fac))
```

    [1] zebra    swing    aardvark
    Levels: zebra swing aardvark


**Sorting a data frame**

bit difficult normally, but use the `sortFrame()` method in lsr package


```R
sortFrame(garden, speaker, line)
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th><th scope=col>line</th></tr></thead>
<tbody>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td><td>7          </td></tr>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td><td>9          </td></tr>
	<tr><th scope=row>case.3</th><td>tombliboo  </td><td>ee         </td><td>5          </td></tr>
	<tr><th scope=row>case.1</th><td>upsy-daisy </td><td>pip        </td><td>1          </td></tr>
	<tr><th scope=row>case.2</th><td>upsy-daisy </td><td>pip        </td><td>2          </td></tr>
</tbody>
</table>



Sorts by speaker, then sorts by line

use minus sign for reverse order


```R
sortFrame(garden, speaker, -line)
```


<table>
<thead><tr><th></th><th scope=col>speaker</th><th scope=col>utterance</th><th scope=col>line</th></tr></thead>
<tbody>
	<tr><th scope=row>case.5</th><td>makka-pakka</td><td>onk        </td><td>9          </td></tr>
	<tr><th scope=row>case.4</th><td>makka-pakka</td><td>pip        </td><td>7          </td></tr>
	<tr><th scope=row>case.3</th><td>tombliboo  </td><td>ee         </td><td>5          </td></tr>
	<tr><th scope=row>case.2</th><td>upsy-daisy </td><td>pip        </td><td>2          </td></tr>
	<tr><th scope=row>case.1</th><td>upsy-daisy </td><td>pip        </td><td>1          </td></tr>
</tbody>
</table>



**Binding vectors together**


```R
cake.1 <- c(100, 80, 0, 0, 0)
cake.2 <- c(100, 100, 90, 30, 10)
```

combine with data frame


```R
cake.df <- data.frame(cake.1, cake.2)
cake.df
```


<table>
<thead><tr><th scope=col>cake.1</th><th scope=col>cake.2</th></tr></thead>
<tbody>
	<tr><td>100</td><td>100</td></tr>
	<tr><td> 80</td><td>100</td></tr>
	<tr><td>  0</td><td> 90</td></tr>
	<tr><td>  0</td><td> 30</td></tr>
	<tr><td>  0</td><td> 10</td></tr>
</tbody>
</table>



column bind (cbind())


```R
cake.mat1 <- cbind(cake.1, cake.2)
print(cake.mat1)
```

         cake.1 cake.2
    [1,]    100    100
    [2,]     80    100
    [3,]      0     90
    [4,]      0     30
    [5,]      0     10


note that this is a matrix, not data.frame

rbind() binds row-wise, not column-wise


```R
cake.mat2 <- rbind(cake.1, cake.2)
print(cake.mat2)
```

           [,1] [,2] [,3] [,4] [,5]
    cake.1  100   80    0    0    0
    cake.2  100  100   90   30   10


can add names using rownames() and colnames(). merge() can do database-like merging of vectors and data frames

**Binding multiple copies of the same vector together**


```R
fibonacci <- c(1, 1, 2, 3, 5, 8)
print(rbind(fibonacci, fibonacci, fibonacci))
```

              [,1] [,2] [,3] [,4] [,5] [,6]
    fibonacci    1    1    2    3    5    8
    fibonacci    1    1    2    3    5    8
    fibonacci    1    1    2    3    5    8


lsr package: rowCopy and colCopy


```R
print(rowCopy(fibonacci, times=3))
```

         [,1] [,2] [,3] [,4] [,5] [,6]
    [1,]    1    1    2    3    5    8
    [2,]    1    1    2    3    5    8
    [3,]    1    1    2    3    5    8


**Transposing a matrix or data frame**


```R
load("cakes.Rdata")
print(cakes)
```

           time.1 time.2 time.3 time.4 time.5
    cake.1    100     80      0      0      0
    cake.2    100    100     90     30     10
    cake.3    100     20     20     20     20
    cake.4    100    100    100    100    100



```R
class(cakes)
```


'matrix'



```R
cakes.flipped <- t(cakes)
print(cakes.flipped)
```

           cake.1 cake.2 cake.3 cake.4
    time.1    100    100    100    100
    time.2     80    100     20    100
    time.3      0     90     20    100
    time.4      0     30     20    100
    time.5      0     10     20    100


use tFrame() from lsr package to transpose dataframes


```R
tFrame(itng)
```


<table>
<thead><tr><th></th><th scope=col>V1</th><th scope=col>V2</th><th scope=col>V3</th><th scope=col>V4</th><th scope=col>V5</th><th scope=col>V6</th><th scope=col>V7</th><th scope=col>V8</th><th scope=col>V9</th><th scope=col>V10</th></tr></thead>
<tbody>
	<tr><th scope=row>speaker</th><td>upsy-daisy </td><td>upsy-daisy </td><td>upsy-daisy </td><td>upsy-daisy </td><td>tombliboo  </td><td>tombliboo  </td><td>makka-pakka</td><td>makka-pakka</td><td>makka-pakka</td><td>makka-pakka</td></tr>
	<tr><th scope=row>utterance</th><td>pip        </td><td>pip        </td><td>onk        </td><td>onk        </td><td>ee         </td><td>oo         </td><td>pip        </td><td>pip        </td><td>onk        </td><td>onk        </td></tr>
</tbody>
</table>



## Reshaping a data frame

**Long form and wide form data**


```R
load("repeated.Rdata")
who()
```


       -- Name --         -- Class --   -- Size --
       age                numeric       11        
       age.breaks         numeric       4         
       age.group          factor        11        
       age.group2         factor        11        
       age.group3         factor        11        
       age.labels         character     3         
       cake.1             numeric       5         
       cake.2             numeric       5         
       cake.df            data.frame    5 x 2     
       cake.mat1          matrix        5 x 2     
       cake.mat2          matrix        2 x 5     
       cakes              matrix        4 x 5     
       cakes.flipped      matrix        5 x 4     
       choice             data.frame    4 x 10    
       df                 data.frame    4 x 1     
       drugs              data.frame    10 x 8    
       fac                factor        3         
       fibonacci          numeric       6         
       garden             data.frame    5 x 3     
       is.MP.speaking     logical       5         
       itng               data.frame    10 x 2    
       itng.table         table         3 x 4     
       likert.centered    numeric       10        
       likert.raw         numeric       10        
       makka.pakka        character     4         
       numbers            numeric       3         
       opinion.dir        numeric       10        
       opinion.strength   numeric       10        
       speaker            character     10        
       speech.by.char     list          3         
       text               character     3         
       tombliboo          character     2         
       upsy.daisy         character     4         
       utterance          character     10        



```R
drugs
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>gender</th><th scope=col>WMC_alcohol</th><th scope=col>WMC_caffeine</th><th scope=col>WMC_no.drug</th><th scope=col>RT_alcohol</th><th scope=col>RT_caffeine</th><th scope=col>RT_no.drug</th></tr></thead>
<tbody>
	<tr><td>1     </td><td>female</td><td>3.7   </td><td>3.7   </td><td>3.9   </td><td>488   </td><td>236   </td><td>371   </td></tr>
	<tr><td>2     </td><td>female</td><td>6.4   </td><td>7.3   </td><td>7.9   </td><td>607   </td><td>376   </td><td>349   </td></tr>
	<tr><td>3     </td><td>female</td><td>4.6   </td><td>7.4   </td><td>7.3   </td><td>643   </td><td>226   </td><td>412   </td></tr>
	<tr><td>4     </td><td>male  </td><td>6.4   </td><td>7.8   </td><td>8.2   </td><td>684   </td><td>206   </td><td>252   </td></tr>
	<tr><td>5     </td><td>female</td><td>4.9   </td><td>5.2   </td><td>7.0   </td><td>593   </td><td>262   </td><td>439   </td></tr>
	<tr><td>6     </td><td>male  </td><td>5.4   </td><td>6.6   </td><td>7.2   </td><td>492   </td><td>230   </td><td>464   </td></tr>
	<tr><td>7     </td><td>male  </td><td>7.9   </td><td>7.9   </td><td>8.9   </td><td>690   </td><td>259   </td><td>327   </td></tr>
	<tr><td>8     </td><td>male  </td><td>4.1   </td><td>5.9   </td><td>4.5   </td><td>486   </td><td>230   </td><td>305   </td></tr>
	<tr><td>9     </td><td>female</td><td>5.2   </td><td>6.2   </td><td>7.2   </td><td>686   </td><td>273   </td><td>327   </td></tr>
	<tr><td>10    </td><td>female</td><td>6.2   </td><td>7.4   </td><td>7.8   </td><td>645   </td><td>240   </td><td>498   </td></tr>
</tbody>
</table>



wide form: each participant is a single row<br>
two vars which are characteristic of subject (id and gender)<br> 
6 variables -> 2 measured variables in 3 testing conditions<br> 
drug type is a **within-subject factor**

**Reshaping with wideToLong()**

if we want a separate row for each testing occasion

wideToLong() is in lsr package. relies format of variable names

`id', 'gender` = **between-subject** variables


```R
drugs.2 <- wideToLong(drugs, within="drug")
drugs.2
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>gender</th><th scope=col>drug</th><th scope=col>WMC</th><th scope=col>RT</th></tr></thead>
<tbody>
	<tr><td>1       </td><td>female  </td><td>alcohol </td><td>3.7     </td><td>488     </td></tr>
	<tr><td>2       </td><td>female  </td><td>alcohol </td><td>6.4     </td><td>607     </td></tr>
	<tr><td>3       </td><td>female  </td><td>alcohol </td><td>4.6     </td><td>643     </td></tr>
	<tr><td>4       </td><td>male    </td><td>alcohol </td><td>6.4     </td><td>684     </td></tr>
	<tr><td>5       </td><td>female  </td><td>alcohol </td><td>4.9     </td><td>593     </td></tr>
	<tr><td>6       </td><td>male    </td><td>alcohol </td><td>5.4     </td><td>492     </td></tr>
	<tr><td>7       </td><td>male    </td><td>alcohol </td><td>7.9     </td><td>690     </td></tr>
	<tr><td>8       </td><td>male    </td><td>alcohol </td><td>4.1     </td><td>486     </td></tr>
	<tr><td>9       </td><td>female  </td><td>alcohol </td><td>5.2     </td><td>686     </td></tr>
	<tr><td>10      </td><td>female  </td><td>alcohol </td><td>6.2     </td><td>645     </td></tr>
	<tr><td>1       </td><td>female  </td><td>caffeine</td><td>3.7     </td><td>236     </td></tr>
	<tr><td>2       </td><td>female  </td><td>caffeine</td><td>7.3     </td><td>376     </td></tr>
	<tr><td>3       </td><td>female  </td><td>caffeine</td><td>7.4     </td><td>226     </td></tr>
	<tr><td>4       </td><td>male    </td><td>caffeine</td><td>7.8     </td><td>206     </td></tr>
	<tr><td>5       </td><td>female  </td><td>caffeine</td><td>5.2     </td><td>262     </td></tr>
	<tr><td>6       </td><td>male    </td><td>caffeine</td><td>6.6     </td><td>230     </td></tr>
	<tr><td>7       </td><td>male    </td><td>caffeine</td><td>7.9     </td><td>259     </td></tr>
	<tr><td>8       </td><td>male    </td><td>caffeine</td><td>5.9     </td><td>230     </td></tr>
	<tr><td>9       </td><td>female  </td><td>caffeine</td><td>6.2     </td><td>273     </td></tr>
	<tr><td>10      </td><td>female  </td><td>caffeine</td><td>7.4     </td><td>240     </td></tr>
	<tr><td>1       </td><td>female  </td><td>no.drug </td><td>3.9     </td><td>371     </td></tr>
	<tr><td>2       </td><td>female  </td><td>no.drug </td><td>7.9     </td><td>349     </td></tr>
	<tr><td>3       </td><td>female  </td><td>no.drug </td><td>7.3     </td><td>412     </td></tr>
	<tr><td>4       </td><td>male    </td><td>no.drug </td><td>8.2     </td><td>252     </td></tr>
	<tr><td>5       </td><td>female  </td><td>no.drug </td><td>7.0     </td><td>439     </td></tr>
	<tr><td>6       </td><td>male    </td><td>no.drug </td><td>7.2     </td><td>464     </td></tr>
	<tr><td>7       </td><td>male    </td><td>no.drug </td><td>8.9     </td><td>327     </td></tr>
	<tr><td>8       </td><td>male    </td><td>no.drug </td><td>4.5     </td><td>305     </td></tr>
	<tr><td>9       </td><td>female  </td><td>no.drug </td><td>7.2     </td><td>327     </td></tr>
	<tr><td>10      </td><td>female  </td><td>no.drug </td><td>7.8     </td><td>498     </td></tr>
</tbody>
</table>



**Reshaping data using longToWide()**

use a formula to indicate which variables are measured separately for each condition, and which is the within-subject factor specifying the condition

2 sided formula: measured vars ~ within-subject factor vars


```R
longToWide(drugs.2, formula = WMC+RT~drug)
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>gender</th><th scope=col>WMC_alcohol</th><th scope=col>RT_alcohol</th><th scope=col>WMC_caffeine</th><th scope=col>RT_caffeine</th><th scope=col>WMC_no.drug</th><th scope=col>RT_no.drug</th></tr></thead>
<tbody>
	<tr><td>1     </td><td>female</td><td>3.7   </td><td>488   </td><td>3.7   </td><td>236   </td><td>3.9   </td><td>371   </td></tr>
	<tr><td>2     </td><td>female</td><td>6.4   </td><td>607   </td><td>7.3   </td><td>376   </td><td>7.9   </td><td>349   </td></tr>
	<tr><td>3     </td><td>female</td><td>4.6   </td><td>643   </td><td>7.4   </td><td>226   </td><td>7.3   </td><td>412   </td></tr>
	<tr><td>4     </td><td>male  </td><td>6.4   </td><td>684   </td><td>7.8   </td><td>206   </td><td>8.2   </td><td>252   </td></tr>
	<tr><td>5     </td><td>female</td><td>4.9   </td><td>593   </td><td>5.2   </td><td>262   </td><td>7.0   </td><td>439   </td></tr>
	<tr><td>6     </td><td>male  </td><td>5.4   </td><td>492   </td><td>6.6   </td><td>230   </td><td>7.2   </td><td>464   </td></tr>
	<tr><td>7     </td><td>male  </td><td>7.9   </td><td>690   </td><td>7.9   </td><td>259   </td><td>8.9   </td><td>327   </td></tr>
	<tr><td>8     </td><td>male  </td><td>4.1   </td><td>486   </td><td>5.9   </td><td>230   </td><td>4.5   </td><td>305   </td></tr>
	<tr><td>9     </td><td>female</td><td>5.2   </td><td>686   </td><td>6.2   </td><td>273   </td><td>7.2   </td><td>327   </td></tr>
	<tr><td>10    </td><td>female</td><td>6.2   </td><td>645   </td><td>7.4   </td><td>240   </td><td>7.8   </td><td>498   </td></tr>
</tbody>
</table>



**Reshaping with multiple within-subject factors**


```R
choice
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>gender</th><th scope=col>MRT/block1/day1</th><th scope=col>MRT/block1/day2</th><th scope=col>MRT/block2/day1</th><th scope=col>MRT/block2/day2</th><th scope=col>PC/block1/day1</th><th scope=col>PC/block1/day2</th><th scope=col>PC/block2/day1</th><th scope=col>PC/block2/day2</th></tr></thead>
<tbody>
	<tr><td>1     </td><td>male  </td><td>415   </td><td>400   </td><td>455   </td><td>450   </td><td>79    </td><td>88    </td><td>82    </td><td> 93   </td></tr>
	<tr><td>2     </td><td>male  </td><td>500   </td><td>490   </td><td>532   </td><td>518   </td><td>83    </td><td>92    </td><td>86    </td><td> 97   </td></tr>
	<tr><td>3     </td><td>female</td><td>478   </td><td>468   </td><td>499   </td><td>474   </td><td>91    </td><td>98    </td><td>90    </td><td>100   </td></tr>
	<tr><td>4     </td><td>female</td><td>550   </td><td>502   </td><td>602   </td><td>588   </td><td>75    </td><td>89    </td><td>78    </td><td> 95   </td></tr>
</tbody>
</table>




```R
choice.2 <- wideToLong(choice, within=c("block", "day"), sep="/")
choice.2
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>gender</th><th scope=col>MRT</th><th scope=col>PC</th><th scope=col>block</th><th scope=col>day</th></tr></thead>
<tbody>
	<tr><td>1     </td><td>male  </td><td>415   </td><td> 79   </td><td>block1</td><td>day1  </td></tr>
	<tr><td>2     </td><td>male  </td><td>500   </td><td> 83   </td><td>block1</td><td>day1  </td></tr>
	<tr><td>3     </td><td>female</td><td>478   </td><td> 91   </td><td>block1</td><td>day1  </td></tr>
	<tr><td>4     </td><td>female</td><td>550   </td><td> 75   </td><td>block1</td><td>day1  </td></tr>
	<tr><td>1     </td><td>male  </td><td>400   </td><td> 88   </td><td>block1</td><td>day2  </td></tr>
	<tr><td>2     </td><td>male  </td><td>490   </td><td> 92   </td><td>block1</td><td>day2  </td></tr>
	<tr><td>3     </td><td>female</td><td>468   </td><td> 98   </td><td>block1</td><td>day2  </td></tr>
	<tr><td>4     </td><td>female</td><td>502   </td><td> 89   </td><td>block1</td><td>day2  </td></tr>
	<tr><td>1     </td><td>male  </td><td>455   </td><td> 82   </td><td>block2</td><td>day1  </td></tr>
	<tr><td>2     </td><td>male  </td><td>532   </td><td> 86   </td><td>block2</td><td>day1  </td></tr>
	<tr><td>3     </td><td>female</td><td>499   </td><td> 90   </td><td>block2</td><td>day1  </td></tr>
	<tr><td>4     </td><td>female</td><td>602   </td><td> 78   </td><td>block2</td><td>day1  </td></tr>
	<tr><td>1     </td><td>male  </td><td>450   </td><td> 93   </td><td>block2</td><td>day2  </td></tr>
	<tr><td>2     </td><td>male  </td><td>518   </td><td> 97   </td><td>block2</td><td>day2  </td></tr>
	<tr><td>3     </td><td>female</td><td>474   </td><td>100   </td><td>block2</td><td>day2  </td></tr>
	<tr><td>4     </td><td>female</td><td>588   </td><td> 95   </td><td>block2</td><td>day2  </td></tr>
</tbody>
</table>




```R
longToWide(choice.2, MRT+PC~block+day, sep="/")
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>gender</th><th scope=col>MRT/block1/day1</th><th scope=col>PC/block1/day1</th><th scope=col>MRT/block1/day2</th><th scope=col>PC/block1/day2</th><th scope=col>MRT/block2/day1</th><th scope=col>PC/block2/day1</th><th scope=col>MRT/block2/day2</th><th scope=col>PC/block2/day2</th></tr></thead>
<tbody>
	<tr><td>1     </td><td>male  </td><td>415   </td><td>79    </td><td>400   </td><td>88    </td><td>455   </td><td>82    </td><td>450   </td><td> 93   </td></tr>
	<tr><td>2     </td><td>male  </td><td>500   </td><td>83    </td><td>490   </td><td>92    </td><td>532   </td><td>86    </td><td>518   </td><td> 97   </td></tr>
	<tr><td>3     </td><td>female</td><td>478   </td><td>91    </td><td>468   </td><td>98    </td><td>499   </td><td>90    </td><td>474   </td><td>100   </td></tr>
	<tr><td>4     </td><td>female</td><td>550   </td><td>75    </td><td>502   </td><td>89    </td><td>602   </td><td>78    </td><td>588   </td><td> 95   </td></tr>
</tbody>
</table>



**What other options are there?**

reshape(), stack(), unstack()<br> 

reshape package -> melt() and cast()<br> 

## Working with Text

**Shortening a string**


```R
animals <- c("cat", "dog", "kangaroo", "whale")
strtrim(animals, width=3)
```


<ol class=list-inline>
	<li>'cat'</li>
	<li>'dog'</li>
	<li>'kan'</li>
	<li>'wha'</li>
</ol>




```R
substr(animals, start=2, stop=3)
```


<ol class=list-inline>
	<li>'at'</li>
	<li>'og'</li>
	<li>'an'</li>
	<li>'ha'</li>
</ol>



**pasting strings together**
paste():<br> 
- \<strings to paste together\>
- sep = seperators (" " by default)
- collapse- whether the inputs should be collapsed. default: None
    


```R
paste("hello", "world")
```


'hello world'



```R
paste("hello", "world", sep=".")
```


'hello.world'



```R
hw <- c("hello", "world")
ng <- c("nasty", "government")
```


```R
paste(hw, ng)
```


<ol class=list-inline>
	<li>'hello nasty'</li>
	<li>'world government'</li>
</ol>




```R
paste(hw, ng, sep=".")
```


<ol class=list-inline>
	<li>'hello.nasty'</li>
	<li>'world.government'</li>
</ol>




```R
paste(hw, ng, collapse=".")
```


'hello nasty.world government'



```R
paste(hw, ng, sep=".", collapse=":::")
```


'hello.nasty:::world.government'


**splitting strings**


```R
monkey <- "It was the best of times. It was the blurst of times."
```

use strsplit()<br>
- x=vector of character strings to be split
- split = fixed string or regular expression
- fixed = fixed delimiter (FALSE by default, should usually be set to true)


```R
monkey.1 <- strsplit(monkey, split=" ", fixed=TRUE)
print(monkey.1)
```

    [[1]]
     [1] "It"     "was"    "the"    "best"   "of"     "times." "It"     "was"   
     [9] "the"    "blurst" "of"     "times."
    


can use unlist function for single inputs


```R
print(unlist(monkey.1))
```

     [1] "It"     "was"    "the"    "best"   "of"     "times." "It"     "was"   
     [9] "the"    "blurst" "of"     "times."


**Making simple conversions**

toupper(), tolower() (does what you think they do)

chartr() - character by character substitution


```R
old.text <- "netflix"
chartr(old=c("e"), new=c("o"), x=old.text)
```


'notflix'


**Applying logical operators to text**

- uppercase letters come before lowercase


```R
"anteater" < "ZEBRA"
```


TRUE


may have been changed in an update or something

**Concatenating and printing with cat()**


```R
cat(hw, ng)
```

    hello world nasty government


```R
paste(hw, ng, collapse=" ")
```


'hello nasty world government'


cat is for printing. It does not return anything


```R
x<-cat(hw, ng)
x
```

    hello world nasty government


    NULL


print will print literally


```R
print("hello\nworld")
```

    [1] "hello\nworld"


cat will interpret special characters


```R
cat("hello\nworld")
```

    hello
    world

**Using escape characters in text**

![image.png](attachment:image.png)


```R
PJ <- "P.J. O\'Rourke says, \"Yay, money!\". It\'s a joke, but no-one laughs."
print(PJ)
```

    [1] "P.J. O'Rourke says, \"Yay, money!\". It's a joke, but no-one laughs."



```R
print.noquote(PJ)
```

    [1] P.J. O'Rourke says, "Yay, money!". It's a joke, but no-one laughs.



```R
cat(PJ)
```

    P.J. O'Rourke says, "Yay, money!". It's a joke, but no-one laughs.

**Matching and substituting text**

grep(), gsub(), and sub()


```R
beers <- c("little creatures", "sierra nevada", "coopers pale")
grep(patter="er", x=beers, fixed=TRUE)
```


<ol class=list-inline>
	<li>2</li>
	<li>3</li>
</ol>



gsub() - replace all instances<br> 
sub() - replaces first instance<br> 


```R
gsub(pattern="a", replacement="BLAH", x=beers, fixed=TRUE)
```


<ol class=list-inline>
	<li>'little creBLAHtures'</li>
	<li>'sierrBLAH nevBLAHdBLAH'</li>
	<li>'coopers pBLAHle'</li>
</ol>




```R
sub(pattern="a", replacement="BLAH", x=beers, fixed=TRUE)
```


<ol class=list-inline>
	<li>'little creBLAHtures'</li>
	<li>'sierrBLAH nevada'</li>
	<li>'coopers pBLAHle'</li>
</ol>



**Regular expressions**

## Reading unusual data files

**Loading data from text files**

**read_csv**<br> 
- header: if the first row does not contain column names, set this to False
- sep: delimeter (usually ",")
- quote: specify which character is used for quotes
- skip: number of lines to ignore
- na.strings: special string to indicate that an entry is missing


```R
data <- read.csv(file="booksales2.csv",
                header=FALSE,
                skip=8,
                quote="*",
                sep="\t",
                na.strings="NFI")
```


```R
head(data)
```


<table>
<thead><tr><th scope=col>V1</th><th scope=col>V2</th><th scope=col>V3</th><th scope=col>V4</th></tr></thead>
<tbody>
	<tr><td>January  </td><td>31       </td><td>  0      </td><td>high     </td></tr>
	<tr><td>February </td><td>28       </td><td>100      </td><td>high     </td></tr>
	<tr><td>March    </td><td>31       </td><td>200      </td><td>low      </td></tr>
	<tr><td>April    </td><td>30       </td><td> 50      </td><td>out      </td></tr>
	<tr><td>May      </td><td>31       </td><td> NA      </td><td>out      </td></tr>
	<tr><td>June     </td><td>30       </td><td>  0      </td><td>high     </td></tr>
</tbody>
</table>



other functions for opening other types of data files (using foreign library)

- read.spss()


library(gdata)<br> 
- read.xls()

library(R.matlab) (MATLAB & Octave)<br> 
- readMat()

ect ect. 

## Coercing data from one class to another


```R
x <- "100"
class(x)
```


'character'



```R
x <- as.numeric(x)
class(x)
```


'numeric'



```R
x+1
```


101



```R
x <- as.character(x)
class(x)
```


'character'



```R
as.numeric("hello world")
```

    Warning message in eval(expr, envir, enclos):
    “NAs introduced by coercion”


&lt;NA&gt;


**for booleans**
- can be coerced to TRUE: "T", "TRUE", "True", "true", 1
- can be coerced to FALSE: "F", "FALSE", "False", "false", 0



## Other useful data structures


**Matricies (and arrays)**


```R
row.1 <- c(2, 3, 1)
row.2 <- c(5, 6, 7)

M <- rbind(row.1, row.2)
print(M)
```

          [,1] [,2] [,3]
    row.1    2    3    1
    row.2    5    6    7



```R
colnames(M) <- c("col.1", "col.2", "col.3")
print(M)
```

          col.1 col.2 col.3
    row.1     2     3     1
    row.2     5     6     7



```R
M[2, 3]
```


7



```R
M[2,]
```


<dl class=dl-horizontal>
	<dt>col.1</dt>
		<dd>5</dd>
	<dt>col.2</dt>
		<dd>6</dd>
	<dt>col.3</dt>
		<dd>7</dd>
</dl>




```R
M[,3]
```


<dl class=dl-horizontal>
	<dt>row.1</dt>
		<dd>1</dd>
	<dt>row.2</dt>
		<dd>7</dd>
</dl>



Note on matrix multiplication: R has no concept of a row vector or column vector, so when doing matrix\*vector, R treates the vector as being in whichever orientation makes the calculation work. 

matricies must be of a homogeneous datatype


```R
class(M[1])
```


'numeric'



```R
M[1, 2] <- "text"
```


```R
M
```


<table>
<thead><tr><th></th><th scope=col>col.1</th><th scope=col>col.2</th><th scope=col>col.3</th></tr></thead>
<tbody>
	<tr><th scope=row>row.1</th><td>2   </td><td>text</td><td>1   </td></tr>
	<tr><th scope=row>row.2</th><td>5   </td><td>6   </td><td>7   </td></tr>
</tbody>
</table>




```R
class(M[1])
```


'character'


**arrays / 3d data structures**


```R
dan.awake <- c(T, T, T, T, T, F, F, F, F, F)
xtab.3d <- table(speaker, utterance, dan.awake)
print(xtab.3d)
```

    , , dan.awake = FALSE
    
                 utterance
    speaker       ee onk oo pip
      makka-pakka  0   2  0   2
      tombliboo    0   0  1   0
      upsy-daisy   0   0  0   0
    
    , , dan.awake = TRUE
    
                 utterance
    speaker       ee onk oo pip
      makka-pakka  0   0  0   0
      tombliboo    1   0  0   0
      upsy-daisy   0   2  0   2
    


**Ordered factors**

2 different types of factors in R: ordered and unordered<br> 

unordered factor = nominal scaled variable


```R
likert.raw
```


<ol class=list-inline>
	<li>1</li>
	<li>7</li>
	<li>3</li>
	<li>4</li>
	<li>4</li>
	<li>4</li>
	<li>2</li>
	<li>6</li>
	<li>5</li>
	<li>5</li>
</ol>




```R
likert.ordinal <- factor(x=likert.raw,
                        levels=seq(7, 1, -1),
                        ordered=TRUE)
print(likert.ordinal)
```

     [1] 1 7 3 4 4 4 2 6 5 5
    Levels: 7 < 6 < 5 < 4 < 3 < 2 < 1


*always ensure that your ordered factors are ordered properly*


```R
levels(likert.ordinal) <- c("strong.disagree", "disagree", "weak.disagree",
                           "neutral", "weak.agree", "agree", "strong.agree")
print(likert.ordinal)
```

     [1] strong.agree    strong.disagree weak.agree      neutral        
     [5] neutral         neutral         agree           disagree       
     [9] weak.disagree   weak.disagree  
    7 Levels: strong.disagree < disagree < weak.disagree < ... < strong.agree


**Dates and times**


```R
print(Sys.Date())
```

    [1] "2021-07-18"



```R
today <- Sys.Date()
print(today+365)
```

    [1] "2022-07-18"


weekdays() - tells you what day of the week a particular day is on


```R
weekdays(today)
```


'Sunday'


## MIscellaneous topics

**Problems with floating point arithmetic**


```R
0.1+0.2==0.3
```


FALSE


There are super small rounding errors that occur when the computer stores these results in memory: 


```R
0.1+0.2-0.3
```


5.55111512312578e-17


decimals like 0.1 are actually very long in binary

**The recycling rule**


```R
x <- c(1, 1, 1, 1, 1, 1)
y <- c(0, 1)
x+y
```


<ol class=list-inline>
	<li>1</li>
	<li>2</li>
	<li>1</li>
	<li>2</li>
	<li>1</li>
	<li>2</li>
</ol>



R recycled the shorter vector several times (i.e. broadcasted?)

**Environments**


```R
search()
```


<ol class=list-inline>
	<li>'.GlobalEnv'</li>
	<li>'package:lsr'</li>
	<li>'jupyter:irkernel'</li>
	<li>'package:stats'</li>
	<li>'package:graphics'</li>
	<li>'package:grDevices'</li>
	<li>'package:utils'</li>
	<li>'package:datasets'</li>
	<li>'package:methods'</li>
	<li>'Autoloads'</li>
	<li>'package:base'</li>
</ol>



**Attaching a data frame**

attach() copies the columns from a dataframe to the workspace
