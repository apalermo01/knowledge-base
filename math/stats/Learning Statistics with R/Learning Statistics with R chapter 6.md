# Drawing Graphs

## An overview of R graphics

**painting analogy**

device- the thing you paint the graphic onto (i.e. window- for Ubuntu it's probably X11, windows: windows, mac: quartz, Rstudio: RStudioGD)

what we're painting with: two different graphics packages: traditional graphics (graphics) and grid graphics (grid). Everything we're doing here uses traditional graphics

painting style: high level commands- many different packages. traditional graphics: graphics packages. grid graphics: lattice and ggplots2

## Intro to plotting


```R
Fibonacci <- c(1, 1, 2, 3, 5, 8, 13)
plot(Fibonacci)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_3_0.png)
    


plot is another generic function


**graphical parameters**<br>
universal, general purpose arguments for basically any plotting function. can be changed using `par()` or by passing the args directly to the plotting function

**Title and axis labels**<br> 

- main: title
- sub: subitlte
- xlab: x-axis label
- ylab: y-axis label


```R
plot(Fibonacci,
      main="specify title with 'main' argument", 
      sub="subtitle here",
      xlab="xlab",
      ylab="ylab")
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_6_0.png)
    


**font styles**

- font.main
- font.sub
- font.lab
- font.axis
can't specify different styles for x and y labels w/ out low-level functions

- 1 = plain text
- 2 = boldface
- 3 = italic
- 4 = bold-italic

**font colours**

- col.main
- col.sub
- col.lab
- col.axis

can use english name of desired color, find options in `colours()` or use rgb() or hsv() to directly select color

**font size**

- cex.main
- cex.sub
- cex.lab
- cex.axis

cex = 'character expansion' - works like magnification. 1 by default, except for title, which is defaulted to 1.2

**font family**
`family` = "sans", "serif", or "mono" or actual font names


```R
plot(Fibonacci,
    main="The first 7 Fibonacci numbers",
    xlab = "Position in the sequence",
    ylab = "The Fibonacci number",
    font.main = 1,
    cex.main = 1,
    font.axis = 2,
    col.lab = "gray50")
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_8_0.png)
    


**plot type**<br> 
`type` - possible values:
- p - points only
- l - draw line
- o - draw line over the top of points
- b - draw points and lines but don't overplot
- h - draw histogram-like vertical bars
- s - staircase (horizontal then vertical)
- S - staircase (vertical then horizontal)
- c - only connecting lines
- n- nothing

![image.png](attachment:image.png)

**changing other features**

- `col` - color of plot
- `pch` - character used for points
- `cex` - plot size
- `lty` - line type (0-7 or character string)
- 'lwd' - line width
![image.png](attachment:image.png)

**axis appearence**

- `xlim`, `ylim` - axis scales
- `ann` - suppress labeling = no title, subtitle, or axis labels (T or F)
- `axes`- if FALSE, remove axes and numbering (but not labels) `xaxt` or `yaxt` to do each axis individually
- `frame.plot` - include framing box


```R
plot(Fibonacci,
    xlim=c(0, 15),
    ylim=c(0,15),
    ann=FALSE,
    axes=FALSE,
    frame.plot=TRUE)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_12_0.png)
    


## Histograms


```R
library(lsr)
load("aflsmall.Rdata")
who()
```


       -- Name --      -- Class --   -- Size --
       afl.finalists   factor        400       
       afl.margins     numeric       176       
       Fibonacci       numeric         7       



```R
hist(afl.margins, cex=0.2)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_15_0.png)
    



```R
hist(afl.margins, breaks=3)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_16_0.png)
    



```R
hist(afl.margins, breaks=0:116)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_17_0.png)
    


## Stem and leaf plots


```R
stem(afl.margins)
```

    
      The decimal point is 1 digit(s) to the right of the |
    
       0 | 001111223333333344567788888999999
       1 | 0000011122234456666899999
       2 | 00011222333445566667788999999
       3 | 01223555566666678888899
       4 | 012334444477788899
       5 | 00002233445556667
       6 | 0113455678
       7 | 01123556
       8 | 122349
       9 | 458
      10 | 148
      11 | 6
    


## Boxplots


```R
boxplot(afl.margins)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_21_0.png)
    



```R
boxplot(afl.margins,
       xlab="AFL games, 2010",
       ylab="Winning Margin",
       border="grey50",
      frame.plot=FALSE,
       staplewex=0,
       whisklty=1)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_22_0.png)
    


**drawing multiple boxplots**


```R
load("aflsmall2.Rdata")
who(TRUE)
```


       -- Name --      -- Class --   -- Size --
       afl.finalists   factor        400       
       afl.margins     numeric       176       
       afl2            data.frame    4296 x 2  
        $margin        numeric       4296      
        $year          numeric       4296      
       Fibonacci       numeric       7         



```R
head(afl2)
```


<table>
<thead><tr><th scope=col>margin</th><th scope=col>year</th></tr></thead>
<tbody>
	<tr><td>33  </td><td>1987</td></tr>
	<tr><td>59  </td><td>1987</td></tr>
	<tr><td>45  </td><td>1987</td></tr>
	<tr><td>91  </td><td>1987</td></tr>
	<tr><td>39  </td><td>1987</td></tr>
	<tr><td> 1  </td><td>1987</td></tr>
</tbody>
</table>




```R
boxplot(formula=margin~year, data=afl2)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_26_0.png)
    



```R
boxplot(formula = margin ~ year,
        data = afl2,
        xlab = "ALF season",
       ylab = "Winning margin",
       frame.plot=FALSE,
       staplewex = 0,
       staplecol="white",
       boxwex=0.75,
       boxfill="grey80",
       whisklty = 1,
       whiskcol="grey70",
       boxcol="grey70",
       outcol="grey70",
        outpch = 20,                
        outcex = .5,            
        medlty = "blank",          
         medpch = 20,               
         medlwd = 1.5                
       )
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_27_0.png)
    


## Scatterplots


```R
load("parenthood.Rdata")
```


```R
plot(x=parenthood$dan.sleep, y=parenthood$dan.grump)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_30_0.png)
    



```R
plot(x=parenthood$dan.sleep,
    y=parenthood$dan.grump,
    xlab="sleep (hours)",
    ylab="grumpiness (0-100)",
    xlim=c(0, 12),
    ylim=c(0, 100),
    pch=20,
    col="gray50",
    frame.plot=FALSE)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_31_0.png)
    


Scatterplot matrix


```R
pairs(formula = ~ dan.sleep + baby.sleep + dan.grump, data=parenthood)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_33_0.png)
    



```R
pairs(parenthood)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_34_0.png)
    


## Bar Graphs


```R
freq <- tabulate(afl.finalists)
print(freq)
```

     [1] 26 25 26 28 32  0  6 39 27 28 28 17  6 24 26 38 24



```R
teams <- levels(afl.finalists)
print(teams)
```

     [1] "Adelaide"         "Brisbane"         "Carlton"          "Collingwood"     
     [5] "Essendon"         "Fitzroy"          "Fremantle"        "Geelong"         
     [9] "Hawthorn"         "Melbourne"        "North Melbourne"  "Port Adelaide"   
    [13] "Richmond"         "St Kilda"         "Sydney"           "West Coast"      
    [17] "Western Bulldogs"



```R
barplot(freq)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_38_0.png)
    



```R
barplot(height=freq, names.arg=teams)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_39_0.png)
    



```R
barplot(height=freq, names.arg=teams, las=2)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_40_0.png)
    


**changing global settings using par()**

`mar` = vector w/ 4 numbers: space at bottom, left, top, right
default: `mar = c(5.1, 4.1, 4.1, 2.1)


```R
par(mar=c(10.1, 4.1, 4.1, 2.1))
```


```R
barplot(height=freq,
       names.arg=teams,
       las=2,
       ylab="Number of finals",
       main="Finals Played, 1987-2010",
       density=10,
       angle=20)
```


    
![png](Learning%20Statistics%20with%20R%20chapter%206_files/Learning%20Statistics%20with%20R%20chapter%206_43_0.png)
    


Reset graphical parameters


```R
par(mar=c(5.1, 4.1, 4.1, 2.1))
```

## Saving image files


```R
dev.list()
```


<strong>png:</strong> 2



```R
dev.print(device=jpeg,
         filename="testimage.jpg",
         width=480,
         height=300)
```


<strong>png:</strong> 2



```R
getwd()
```


'/home/alex/Documents/GitHub/Data-science-notes/textbooks/Learning Statistics with R'

