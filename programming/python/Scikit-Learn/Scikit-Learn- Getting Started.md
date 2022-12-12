# Getting started ([source](https://scikit-learn.org/stable/getting_started.html#))

Review of the basic functionality of scikit-learn

# Fitting and predicting: Estimator basics

**Estimator** - built in machine learning algorithms / models<br> 
Each estimator has a fit() method to fit data. 

Example with random forest classifier: 


```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)

# Declare a small dataset with 2 samples and 3 features
X = [[1, 2, 3], 
     [11, 12, 13]]
y = [0, 1]
clf.fit(X, y)
```




    RandomForestClassifier(random_state=0)



fit method usually takes 2 inputs: matrix of features (n_samples x n_features), and target values y (regression: real numbers; classification: integers).<br> 

after fitting, predict new values:


```python
clf.predict(X)
```




    array([0, 1])




```python
clf.predict([[4, 5, 6], [14, 15, 16]])
```




    array([0, 1])



# Transformers and pre-processors

preprocessors and transformers have most of the same methods as estimators. Transformers have no predict method, instead they have transform() which returns the new sample matrix. ColumnTransformer can apply different transformations to different features. 


```python
from sklearn.preprocessing import StandardScaler
X = [[0, 15], 
     [1, -10]]

# Scale data according to computed scaling values
StandardScaler().fit(X).transform(X)
```




    array([[-1.,  1.],
           [ 1., -1.]])



# Pipelines: chining pre-processors and estimators

Combine transformers and estimators into a pipeline (same API as the estimator). Prevents data leakage


```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# load iris and split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit entire pipeline
pipe.fit(X_train, y_train)
```




    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('logisticregression', LogisticRegression())])




```python
# now use it like any estimator
accuracy_score(pipe.predict(X_test), y_test)
```




    0.9736842105263158



# Model Evaluation

Perform a 5-fold cross-validation


```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

X, y = make_regression(n_samples=1000, random_state=0)

lr = LinearRegression()

result = cross_validate(lr, X, y)
result['test_score']
```




    array([1., 1., 1., 1., 1.])



# Automatic parameter searches

All estimators have tunable parameters (*hyperparameters*). Automatically find the best parameter combinations using cross-validation. 


```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint

X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# define the parameter space
param_distributions = {'n_estimators': randint(1, 5), 
                       'max_depth': randint(5, 10)}

# create a searchCV object and fit 
search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), 
                           n_iter=5, 
                           param_distributions=param_distributions, 
                           random_state=0)
search.fit(X_train, y_train)
```




    RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter=5,
                       param_distributions={'max_depth': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001810FAEDB88>,
                                            'n_estimators': <scipy.stats._distn_infrastructure.rv_frozen object at 0x000001810FB38408>},
                       random_state=0)




```python
search.best_params_
```




    {'max_depth': 9, 'n_estimators': 4}




```python
# search now acts like a normal estimator with the best parameters
search.score(X_test, y_test)
```




    0.735363411343253



**NOTE**: it's always best to search over a pipeline, not a single estimator. If you preprocess the entire dataset at once, some info about the training set may leak into the test set, violating the assumption that the sets are independent. ([related kaggle post](https://www.kaggle.com/alexisbcook/data-leakage))
