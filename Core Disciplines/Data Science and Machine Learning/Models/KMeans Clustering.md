---
tags:
  - machine_learning
  - unsupervised
  - clustering
---
- See the implementation [[---|Here]] #todo - get link to classical model zoo repo


KMeans clustering is an [[Unsupervised]] algorithm that finds clusters in unlabeled data.

## Formulation
Consider a sequence of observations $(\textbf x_1, \textbf x_2, ..., \textbf x_n)$ where each $\textbf x_i$ is a vector with d-dimensions. k-means clustering partitions the dataset into k sets to that minimized the within-group sum of squares:

$$
\text{arg} \text{min}_S \sum_{i=1}^k \sum_{x \in S_i} ||\textbf x - \mu_i ||^2
$$


## The algorithm

1. Initialize k centroids. This can be done using several different techniques such as:
	1. picking datapoints at random
	2. pick from a uniform distribution based on the spread of each dimension, or restricted on the quantiles (for example- $x_1$ for each centroid ~ uniform distribution with low=10% percentile on that coordinate, high = 90% percentile on that coordinate)
	3. k++ algorithm which aims to optimize centroid initialization
2. For every point in the dataset, calculate its distance to each centroid
3. Assign each datapoint to a centroid
4. Find the average of each of the datapoints associated with each centroid, update the centroid to this new location
5. If any of the centroids changed, repeat steps 2-4, otherwise we've converged on a solution and we can stop

Note that while the optimization condition above is based on the $l_2$ norm (euclidian distance between each point), there's also a variant called K-medians clustering based on the $l_1$ norm where the new centroid is the *median* of the assigned datapoints. 

## References
- https://en.wikipedia.org/wiki/K-means_clustering
- https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/