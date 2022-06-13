# Basic programming

save scripts with .R extension

run using source()


```R
source("hello.R")
```

    [1] "Hello World!"


best practice is to use require() to load a package in a script

### While loops

`while ( CONDITION ) {
    STATEMENT 1
    STATEMENT 2
    ETC
}`


```R
x <- 0

while( x < 1000 ) {
  x <- x + 17
} 
print(x)
```

    [1] 1003


### For loops

`for ( VAR in VECTOR ) {
    STATEMENT1
    STATEMENT2
    ETC
}`


```R
for ( i in 1:3 ) {
  print("Hello")
}
```

    [1] "Hello"
    [1] "Hello"
    [1] "Hello"



```R
words <- c("it", "was", "the", "dirty", "end", "of", "winter")

for (w in words){
  x.length <- nchar(w)
  W <- toupper(w)
  msg <- paste(W, "has", x.length, "letters", sep=" ")
  print(msg)
}
```

    [1] "IT has 2 letters"
    [1] "WAS has 3 letters"
    [1] "THE has 3 letters"
    [1] "DIRTY has 5 letters"
    [1] "END has 3 letters"
    [1] "OF has 2 letters"
    [1] "WINTER has 6 letters"


## conditional statements

`if ( CONDITION ) {
    STATEMENT1
    STATEMENT2
    ETC
} else{
    STATEMENT3
    STATEMENT4
    ETC
}`


```R
today <- Sys.Date()
day <- weekdays(today)

if (day=="Monday"){
  print("I don't like Mondays")
} else {
  pirnt("I'm a happy little automaton")
}
```

    [1] "I don't like Mondays"


can also use `ifelse()` and `switch()`

## Writing functions

`FNAME <- function(ARG1, ARG2, ETC){
    STATEMENT1
    STATEMENT2
    ETC
    return(VALUE)
}`


```R
quadruple <- function(x) {
  y <- x*4
  return(y)
}

class(quadruple)
```


'function'



```R
quadruple(4)
```


16



```R
quadruple
```


<pre class=language-r><code>function (x) 
{
<span style=white-space:pre-wrap>    y &lt;- x * 4</span>
<span style=white-space:pre-wrap>    return(y)</span>
}</code></pre>


Follows the same rules as python for default arguments

... = mechanism to allow users to enter as many inputs as they like


```R
doubleMax <- function(...){
  max.val <- max(...)
  out <- 2*max.val
  return(out)
}

doubleMax(2, 3, 6)
```


12


assign() - create variables in other environments<br> 
functionas can be assigned as elements of a list<br> 

## Implicit loops


```R
words <- c("along", "the", "loom", "of", "the", "land")
sapply(X=words, FUN=nchar)
```


<dl class=dl-horizontal>
	<dt>along</dt>
		<dd>5</dd>
	<dt>the</dt>
		<dd>3</dd>
	<dt>loom</dt>
		<dd>4</dd>
	<dt>of</dt>
		<dd>2</dd>
	<dt>the</dt>
		<dd>3</dd>
	<dt>land</dt>
		<dd>4</dd>
</dl>



`tapply()` - loop over all different values that appear in INDEX. each value defines a group: tapply() makes a subset of X that corresponds to that group and applies FUN to them


```R
load("nightgarden.Rdata")
```


```R
gender <- c("male", "male", "female", "female", "male")
age <- c(10, 12, 9, 11, 13)
tapply(X=age, INDEX=gender, FUN=mean)
```


<dl class=dl-horizontal>
	<dt>female</dt>
		<dd>10</dd>
	<dt>male</dt>
		<dd>11.6666666666667</dd>
</dl>



by() does something very similar but prints the output differently


```R
by(data=age, INDICES=gender, FUN=mean)
```


    gender: female
    [1] 10
    ------------------------------------------------------------ 
    gender: male
    [1] 11.66667


other similar functions: 
- `lapply()`
- `mapply()`
- `apply()`
- `vapply()`
- `rapply()`
- `eapply()`
