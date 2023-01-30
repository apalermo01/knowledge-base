
**SELECT**
specifies which columns to pull

**FROM**
the database you should pull the columns from

**JOIN**
combines multiple tables together

**WHERE**
- filters records based on a condition

**GROUPBY**
- groups rows that have the same value and aggregates the other columns

**ORDER BY**
- sorts records by a specific column

**HAVING**
- same as where, but meant to be applied to aggregated columns

**WITH**
- subquery - let's you treat the results of a query as another table

**DISTINCT**
- used in SELECT clause to return only the unique values

**UNIQUE**
- a constraint in the create table statement to ensure that no value im the column is repeated

**UNION**
- combines the result of 2 or more SELECT statements
	- results from each select statement must have the same number of columns, datatypes, and must appear in the same order
- concatentates results by column

**UNION ALL

### WHERE vs HAVING
- Where: applied on the record-level
- Having: applied on the aggregate-level
	- As a rule of thumb: HAVING comes after GROUPBY, WITH comes before

## Distinct vs. Unique
- Distinct: used in select statements
- Unique: used as a constraint in table creation

## Union vs. Join
- join: concatenates columns
- union: concatenates rows


## References
- https://www.educba.com/sql-clauses/
- https://learnsql.com/blog/sql-having-vs-where/
- https://stackoverflow.com/questions/9253244/sql-having-vs-where
- https://www.postgresql.org/docs/current/tutorial-concepts.html
- https://www.geeksforgeeks.org/sql-with-clause/
- w3schools.com (multiple pages)