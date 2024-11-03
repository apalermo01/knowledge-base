

## Feature scaling

**standardization** - scale data to have mean 0 and standard deviation 1
**normalization** - between 0 and 1

How to choose? 
- standardize if you know the data is not normal but the model expects a normal distribution
- normalize if you know the data is already normal
- Scaling is required for models that use gradient descent, l1/l2 norm, and anything distance-based

# References

- https://www.geeksforgeeks.org/normalization-vs-standardization/
- https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf
- https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/