

# Possible data types

- **Nominal / Categorical**
	- No relationship between possibilities
	- No sense of "bigger" or "smaller"
	- ex: color, gender

- **Ordinal**
	- there's a meaningful way to order the possible values
	- ex: finishing position in a race, degree of agreement or disagreement
	- still no sense of averaging these

- **Interval**
	- differences between these numbers are meaningful, but there's no inherit sense of "zero"
	- ex: temperature (Celsius)\*, year (2008, 2009, etc.)

- **Ratio**
	- now zero really does mean something and we can multiply and divide
	- ex: response time

- **datetime**
	- dates / timestamps
	- timeseries data


# Data preprocessing steps

**Continuous data**

- [[Outlier detection]]
	- check for datapoints that are significantly different from the group. Use tools like [[Z-score]], [[Inter-quartile Range|IQR]], or [[box plots]]
- Re-scaling
	- [[normalization]] / min-max scaling (get the data between 0 and 1)
		- use this when:
		- you know the bounds of the input variables (e.g. 0-255)
		- the output of a model needs to be in a specific range (e.g. neural networks)
		- don't want to make any assumptions about the data being gaussian
	- [[standardization]] (mean 0 std 1, Z-score)
		- use this when:
		- the algorithm assumes a normal distribution
		- when the data has outliers (it could heavily skew normalization)
		- [[PCA]], [[Kmeans|KMeans Clustering]], and [[Support Vector Machines|SVM]] might perform better
	- [[Robust scaling]]
		- similar to normalization / standardization but use median and interquartile range (robust to outliers)
	- Clipping
		- set maximum / minimum threshold - any datapoints beyond that limit are clipped to the extreme
- Discretization
	- binning continuous data into discrete ranges. May be useful for [[Decision Trees|decision trees]]

- Transformation
	- Fix large skews by taking the square root, log, etc.


**Discrete / nominal / categorical**

- Binning
	- Group data into bins if there are many categories
- One-hot encoding
	- represent the categories by a binary sequence. For example, if the categories are red, green, or blue, then red = [0, 0, 1], green = [0, 1, 0], and blue = [1, 0, 0]
	- a variation on this is dummy variable encoding where [0, 0] is a separate category
- label encoding
	- Encode nominal / categorical variables as an integer, so red=0, blue=1, green=2, etc.
	- According to the documentation on scikit-learn, this should only be used on the target variables (y)
- ordinal encoding
	- encode ordinal variables in accordance with their ordering, so low=0, medium=1, high=2. Normally, you would want to pass a list of the categories in the order you want them (if you use scikit-learn)

### Other tools

- Imputation
	- methods for filling in missing values. You can:
		- fill with a constant value
		- drop all rows with nans
		- fill values with aggregate values like mean / median / mode
		- fill with an aggregate value conditioned on other columns (e.g. if x2 has a null when x1 is "red", then fill in the mean of x2 when x1 is also "red")
		- use more sophisticated machine learning / deep learning methods
	
- feature engineering
	- Generate new columns by combining existing columns in different ways. This is largely based on domain knolwedge.

- Dimensionality reduction
	- Principal component analysis
		- represent the dataset in a lower-dimensional space while maintaining the global variance of the dataset. See the pdf on principal component analysis under machine learning for more detail
	- t-sne
		- t-distributed stochastic neighbor embedding
		- primarily for visualization
		- stochastic algorithm - each run may give different results and highly sensitive to choice of parameters
		- designed to retain the local variance between each datapoint




# References

- Learning Statistics with R chapter 2
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
- https://towardsdatascience.com/what-why-and-how-of-t-sne-1f78d13e224d
- https://www.statology.org/dummy-variables-regression/
- Conversations with ChatGPT