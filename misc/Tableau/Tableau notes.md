# Context Filters

- filters, by default, access all rows in the dataset regardless of other filters
- Use a context filter to filter what other filters can see
- esp. useful for top and bottom filters
- ex: filtering top 10 products then filtering by region - does not by default show top 10 products in each region. Filter region first by adding to context THEN filter top 10

## Splitting Data Fields

- ex: split a customer name into separate fields for first and last name
- in data source, right click on column & look at the menus
- split from data pane: right click on field -> transform

## Aggregating Dimensions

- 3 situations when dimensions might be aggregated
- blending multiple data sources 
- aggregate dimension in calculations since aggregate and non-aggregate data can't be mixed
- 

### Different types of aggregations

- 1) in calculated field (another lesson)<br>
- 2) right click and drag into view -> drop field box<br>
- 3) affter adding to view with context menu- right click- hover over 'measure' to see aggregations [doc](https://kb.tableau.com/articles/howto/when-to-use-the-attribute-attr-function)<br>
    - 'attribute' - displays \*, behaves similarly to min and max, useful with multiple data sources<br> 
    - 'minimum' - returns minimum value (top of list = minimum) - still a dimension<br>
    - 'maximum' - returns maximum value (bottom of list = maximum) - still a dimension<br>
    - 'count' - returns number of entries - turns to a temporary measure so it can perform calculations<br>
    - 'count distinct' - returns number of unique entries - turns into a temporary measure<br>
    
### Aggregating dimensions in calculations
ex: <br>
IF SUM(\[Profilt\]) > 0 THEN "profitable" + MIN(\[Category\]) # using minimum since we just want the first (and only) value<br>
ELSE "unprofitable" + MIN(\[category\])

Any calculation that includes an aggregation is a measure

## Specifying Scope and Direction

table calculation: based on data in view- update when view is filtered
1) Quick table calculation<br>
2) Use table calculation dialog box<br>
3) use calculation editor<br>

Focus on \#2<br>
options for scope: table, pane, or cell<br>
direction: across or down<br>
automaticall shows scope and direction on crosstab<br>
cell scope: useful for stacked bars<br>
<br>
what about null values in calculations? it depends<br>
changing context in table calculation will change the output<br>

**order in which filters are applied**<br>
Extract<br>
Data source<br>
context<br>
measure / dimension<br>
table calculations<br>
table calculation filters<br>

right click on field in view -> add table calculation -> opens dialogue box

## Level of Detail Expressions

LOD - retain numbers at various levels of detail<br>
ex: {SUM([profit])} - aggregates profit to the most aggregated leve<br>

3 keywords: <br>
fixed - {FIXED [Region]: SUM([profit])} - fixes aggregation to region<br>
syntax: use {}- most aggregated by default<br>
fix level of detail: {FIXED[Dimension]:<aggregate fucntion>}<br>
    
include & exclude (later)<br>
    
**TODO** play with this more

## Using LOD Expressions with Filters

Order of filters: <br>
1) Extract filters<br>
2) Data source filters<br>
3) Context filters<br>
4) FIXED<br>
5) Dimension filters<br>
6) Include / Exclude<br>
7) Measure Filters<br>
8) table calculation filters<br>
9) Table calculations<br>


## Using Parameters to Give Control

works like a variable<br>

adjust view according to needs<br>
can adjust filter or reference line<br>

3 steps: <br>
Make it<br>
Use it - within filter or calculation<br>
Show it - show controls so user can adjust it<br>

2 ways to use a parameter: in a calculation or with a filter

# Adding parameters to a view

# Using a parameter to swap measures

- create a parameter
- datatype=integer
- allowable values -> list
- value = 1, 2, 3, ...
- Display as: text displayed for that value (i.e. the different measures)

- use the parameter in a case statement<br>
CASE [name of parameter]<br>
WHEN 1 THEN [measure 1]<br>
WHEN 2 THEN [mesasure 2]<br>
.<br>
.<br>
.<br>
END<br>

- build the view
- put the dynamic measure (calculated measure) into the view
- put the control in the view



# Advanced Mapping: Modifying Locations

## Edit errors in geocode data

'ambiguous' - location data is not unique (e.g. multiple counties with the same name in different states)<br>
    - can manually adjust role, type lat or long, or ensure that the state / province is fixed
   
## Assign locations to data
- select 'unkonwn box' -> 'Edit locations' -> matching locations might be 'unrecognized'; can type in city names or specific lat / long of a desired location

# Advanced mapping: customizing tableau's geocoding

you can do custom geocodes by mapping street addresses to lat / long<br>
- join second datasource with lat / long

### batch geocoding (many websites available that can do this)

# Advanced Mapping: Using a Background image

**to use image as background**<br>
world map: each point is lat / long<br>
what about an offic map? - define using x and y values (lat=x, long=y)<br>

Define boundaries: x: 0-99, y: 0-99, or use pixel dimensions<br>
can define however you like, but two above are the most common<br>

- you must have at least one set of x,y values for the background to appear<br>
X-> rows; y -> columns<br>
fit -> entire view<br>
remove axes<br>

Map -> background images -> upload image<br>
follow prompts to calibrate X and Y coordinates<br>

how to find x and y coords for an arbitrary spot: <br>
right click -> annotate. move around annotation and get x and y coords of the given spot<br>

# Viewing distributions

**Building histograms**<br>
select measure -> showMe -> histogram thumbnail<br>
Tableau automatically creates bin sizes<br>

Manual process:<br>
measure -> right click -> create -> bins<br> 
specify bin size<br> 
bin is defined by and inclusive of its lower limit, exclusive of upper limit<br>
This creates a dimension<br> 
binned fields have histogram icon<br> 
labels: by default: lower limit<br>
make age bin a continuous field<br>

fix tick marks: right click x -> edit Axis -> tick marks -> click fixed, select size to match bin size<br>

**Building Box and Whisker Plots**<br>
ex: daily temperatures for every month<br>
temp -> rows<br>
date -> cols -> expand to month<br>
date -> detail -> exact date<br>
use circle mark type -> analytics pane<br>
summarize -> box plot -> drop on cell<br>

can also use the 'show me' menu

# Comparing measures against a goal

# Building Bar-in-Bar charts

- useful for comparing two measures<br>
drop second measure to axis where 2 bars appear -> side-by-side bars<br>
 

 **Measure Names** - dimension with each label in source
 
 **Measure Values** - contains numerical values for each field
 
 Combine bars 
 measure name from rows to color -> get stacked bars
 unstack marks
 analysis menu -> stack marks -> off
 separate overlapping bars
 measure name / measure values determine which measure is on top
 drop down menu on measure name -> change formatting
 on marks card: measure names + ctrl -> size 
 
 # Bullet Graphs
 
 - can show progression towards a goal<br>
reference measure -> detail<br>
right click axis -> reference line -> distribution -> per cell 

# Showing Statistics and Forcasting: Use the Analytics Pane and Trend Lines


Analytics pane -> trendlines - added per pane, one trendline per color<br>
bold line = actual trendline; light lines = confidence bands (95% by default)<br>
R squared - how well trend fits the data<br> 
p-value - significance (generally look for p<0.05 to look for statistical significance

# Generating a Forecast in the view

Requirements: <br>
- continuous date
- 1 measure
- at least 5 data points
- at least 2 seasons if seasonal forcasting

Analytics pan -> drag forcast into view
- tableau automatically picks the highest quality forecast. Underlying model and smoothing coefficients can't be changed
- uses exponential smoothing

# Advanced Dashboards: Using Design Techniques and Filter Actions

**Concept: Planning for a Successful Dashboard**<br>
4 key steps: <br>
1) determine purpose and audience<br>
Why are you making it? <br>
What questions are you answering?<br>
<br>

2) plan dashboard<br>
How will it be viewed and shared? <br>
on tableau public or server? or in a pdf? <br>
Consider creating custom designs for desktop / phone / tablet<br>
Decide on chart types and data<br>
Organize views in a logical way<br>
is it essential / relevant? <br>


<br>
3) build using design best practices<br>
How do you want users to interact with the data?<br>
Streamline use of legends and filters<br>
use actions / parameters to limit the number of filters<br>
follow left-to-right, top-to-bottom order<br>
use limited # of colors and keep them consistent<br>

4) test<br>
In 5 seconds, can a user get the dashboard's purpose & how to use it?<br> 
Check performance<br>
User performance recorder on help bar<br> 

## Using design best practices
**Instructions**<br>
Write thorough instructions to document interactivity<br>
Do not make your useres guess<br> 
<br>
**Tooltips**<br>

**Color**<br>
Use color to draw attention to salent or contrasting data<br>
**Arrangement**<br>
Place most important components in the upper left<br>
**Fonts**<br>
keep it consistent

## Using advanced filter actions
dashboard -> actions -> add action -> filter
