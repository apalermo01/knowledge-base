

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


# Data preprocessing steps

**Continuous data**

- Outlier detection
	- check for datapoints that are significantly different from the group. Use tools like Z-score, IQR, or box plots
- Re-scaling
	- normalization / min-max scaling (get the data between 0 and 1)
		- use this when:
		- you know the bounds of the input variables (e.g. 0-255)
		- the output of a model needs to be in a specific range (e.g. neural networks)
		- don't want to make any assumptions about the data being gaussian
	- standardization (mean 0 std 1, Z-score)
		- use this when:
		- the algorithm assumes a normal distribution
		- when the data has outliers (it could heavily skew normalization)
		- PCA, Kmeans, and SVM might perform better

- Transformation
	- Fix large skews by taking the square root, log, etc.


**Discrete / nominal / categorical**

- Binning
	- Group data into bins if there are many categories
- One-hot encoding
	- represent the categories by a binary sequence. For example, if the categories are red, green, or blue, then red = [0, 0, 1], green = [0, 1, 0], and blue = [1, 0, 0]
- label encoding
	- Encode ordinal variables with values corresponding to their order




\* I got this from a psychology book

# References

- Learning Statistics with R chapter 2
- Conversations with ChatGPT