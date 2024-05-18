**Get the column from a filtered table**
todays_week = CALCULATETABLE(VALUES('date'[week]), 'date'[Date] = TODAY())

**Get the sum from a filtered table**
= SUMX(FILTER(InternetSales, InternetSales[SalesTerritoryID]=5),[Freight])