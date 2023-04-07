# Numpy for beginners

https://numpy.org/doc/1.21/user/absolute_beginners.html

Numpy is the basis for most numerical calculations in python. Numpy arrays are much more efficient than python lists and has a mechanism of specifying data types.

**Array** - grid of values with homogeneous data types<br> 

**rank** - number of dimensions

**shape** - tuple of integers giving the size of the array along each dimension

one way to make an array is from a python list


```python
import numpy as np
a = np.array([1, 2, 3, 4, 5, 6])
b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
```

access element using square brackets


```python
b[0]
```




    array([1, 2, 3, 4])



**ndarray** - array with any number of dimensions<br> 

**vector** - array with one dimension<br> 

**matrix** - 2d array

**tensor** - common name for 3+ dimensional arrays

**axes** - another name for dimensions

**attributes** - information intrinsic to the array

**Methods of creating arrays:**


```python
np.array([1, 2, 3])
```




    array([1, 2, 3])




```python
np.zeros((3, 2))
```




    array([[0., 0.],
           [0., 0.],
           [0., 0.]])




```python
np.ones((3, 2))
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.]])




```python

np.empty((3, 2))
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.]])



note: np.empty is faster than np.zeros but with random initializations

arrays using range of elements


```python
np.arange(4)
```




    array([0, 1, 2, 3])




```python
np.arange(0, 4, 2) # (start, stop, step size)
```




    array([0, 2])



linearly spaced numbers


```python
np.linspace(0, 10, num=5)
```




    array([ 0. ,  2.5,  5. ,  7.5, 10. ])



can specify data type


```python
np.ones(2, dtype=np.int64)
```




    array([1, 1])



## Adding, removing, and sorting elements


```python
a = np.array([23, 5, 2,7 ,434, 67,34, 345])
np.sort(a)
```




    array([  2,   5,   7,  23,  34,  67, 345, 434])



other functions: <br> 
- argsort: sort along specified axis
- lexsort: indirect stable sort on multple keys
- searchsorted: find elements in a sorted array
- partition: partial sort

concatenating arrays


```python
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

np.concatenate((a, b))
```




    array([1, 2, 3, 4, 5, 6, 7, 8])




```python
x = np.array([[1, 2], [3, 4]])
y = np.array([[5, 6], [7, 8]])

np.concatenate((x, y), axis=0)
```




    array([[1, 2],
           [3, 4],
           [5, 6],
           [7, 8]])




```python
np.concatenate((x, y), axis=1)
```




    array([[1, 2, 5, 6],
           [3, 4, 7, 8]])



## Size and shape of an array


```python
arr_example = np.array([[[0, 1, 2, 3],
                         [4, 5, 6, 7]],
                       
                       [[0, 1, 2, 3],
                       [4, 5, 6, 7]],
                       
                       [[0, 1, 2, 3],
                       [4, 5, 6, 7]]])
```


```python
arr_example.ndim
```




    3




```python
arr_example.size
```




    24




```python
arr_example.shape
```




    (3, 2, 4)



## reshape arrays


```python
a = np.arange(6)
print(a)

b = a.reshape(3, 2)
print(b)
```

    [0 1 2 3 4 5]
    [[0 1]
     [2 3]
     [4 5]]



```python
a.reshape(2, 3)
```




    array([[0, 1, 2],
           [3, 4, 5]])




```python
np.reshape(a, newshape=(1, 6), order="C") # a = old array; newshape = shape you want; order = read/write using C-like index order
```




    array([[0, 1, 2, 3, 4, 5]])



## Convert 1d arrays to 2d and add new axis


```python
a = np.array([1, 2, 3, 4, 5, 6])
a.shape
```




    (6,)




```python
row_vector = a[np.newaxis, :]
row_vector.shape # row-vector
```




    (1, 6)




```python
col_vector = a[:, np.newaxis]
col_vector.shape
```




    (6, 1)



can also use np.expand_dims


```python
b = np.expand_dims(a, axis=0)
b.shape
```




    (1, 6)



## Indexing and slicing


```python
data = np.array([1,2,3])
data[1]
```




    2




```python
data[0:2]
```




    array([1, 2])




```python
data[1:]
```




    array([2, 3])




```python
data[-2:]
```




    array([2, 3])




```python
a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[a<5])
```

    [1 2 3 4]


can use booleans to make a new array


```python
five_up = (a>=5)
print(a[five_up])
```

    [ 5  6  7  8  9 10 11 12]



```python
c = a[(a>2) & (a<11)]
c
```




    array([ 3,  4,  5,  6,  7,  8,  9, 10])



can also use **np.nonzero()** to get the indices


```python
b = np.nonzero(a<5)
b
```




    (array([0, 0, 0, 0]), array([0, 1, 2, 3]))



np.nonzero returned a tuple of arrays: one for each dimension


```python
coords = list(zip(b[0], b[1]))
for pt in coords:
  print(pt)
```

    (0, 0)
    (0, 1)
    (0, 2)
    (0, 3)



```python
a[b]
```




    array([1, 2, 3, 4])



Empty array if desired element is not there


```python
np.nonzero(a==42)
```




    (array([], dtype=int64), array([], dtype=int64))



## Create array from existing data


```python
a = np.array([1,2,3,4,5,6,7,8,9,19])
```


```python
arr1 = a[3:8]
arr1
```




    array([4, 5, 6, 7, 8])




```python
a1 = np.array([[1,1],[2,2]])
a2 = np.array([[3,3],[4,4]])
```

vstack


```python
np.vstack((a1, a2))
```




    array([[1, 1],
           [2, 2],
           [3, 3],
           [4, 4]])



hstack


```python
np.hstack((a1, a2))
```




    array([[1, 1, 3, 3],
           [2, 2, 4, 4]])



dstack


```python
np.dstack((a1, a2))
```




    array([[[1, 3],
            [1, 3]],
    
           [[2, 4],
            [2, 4]]])



**hsplit()** - specify number of equally shaped arrays to return or the columns after which division should occur


```python
x = np.arange(1, 25).reshape(2, 12)
np.hsplit(x, 3)
```




    [array([[ 1,  2,  3,  4],
            [13, 14, 15, 16]]),
     array([[ 5,  6,  7,  8],
            [17, 18, 19, 20]]),
     array([[ 9, 10, 11, 12],
            [21, 22, 23, 24]])]




```python
np.hsplit(x, (3, 4))
```




    [array([[ 1,  2,  3],
            [13, 14, 15]]),
     array([[ 4],
            [16]]),
     array([[ 5,  6,  7,  8,  9, 10, 11, 12],
            [17, 18, 19, 20, 21, 22, 23, 24]])]



use a view to make a shallow copy (i.e. changing the copy changes the original)


```python
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
```


```python
b1 = a[0, :]
b1
```




    array([1, 2, 3, 4])




```python
b1[0] = 99
b1
```




    array([99,  2,  3,  4])




```python
a
```




    array([[99,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])



using .copy() will make a deep copy

## Basic array operations

can do mathematical operations between arrays


```python
a = np.array([1, 2])
b = np.ones(2, dtype=int)
a+b
```




    array([2, 3])




```python
b = np.array([[1, 1], [2, 2]])
b.sum()
```




    6




```python
b.sum(axis=0)
```




    array([3, 3])




```python
b.sum(axis=1)
```




    array([2, 4])



## Broadcasting

can perform vector-scalar operations and operations between vectors of different sizes


```python
data = np.array([1.0, 2.0])
data*1.6
```




    array([1.6, 3.2])



## more useful array operations


```python
data.max()
```




    2.0




```python
data.min()
```




    1.0




```python
data.sum()
```




    3.0




```python
a = np.array([[0.45053314, 0.17296777, 0.34376245, 0.5510652],
             [0.54627315, 0.05093587, 0.40067661, 0.55645993],
             [0.12697628, 0.82485143, 0.26590556, 0.56917101]])
a.sum()
```




    4.8595784




```python
a.min()
```




    0.05093587




```python
a.min(axis=0)
```




    array([0.12697628, 0.05093587, 0.26590556, 0.5510652 ])



values correspond to each column

## Creating matricies


```python
data = np.array([[1, 2], [3, 4], [5, 6]])
data
```




    array([[1, 2],
           [3, 4],
           [5, 6]])



slicing


```python
data[0, 1]
```




    2




```python
data[1:3]
```




    array([[3, 4],
           [5, 6]])




```python
data[0:2, 0]
```




    array([1, 3])



aggregations


```python
data.max()
```




    6




```python
data.min()
```




    1




```python
data.sum()
```




    21



and row or column-wise


```python
data.max(axis=0)
```




    array([5, 6])




```python
data.max(axis=1)
```




    array([2, 4, 6])



can do operations with arrays


```python
data = np.array([[1, 2], [3, 4]])
ones = np.array([[1, 1], [1, 1]])

data+ones
```




    array([[2, 3],
           [4, 5]])



can broadcast when one matrix has only one column or row


```python
data = np.array([[1, 2], [3, 4], [5, 6]])
ones_row = np.array([[1, 1]])
data+ones_row
```




    array([[2, 3],
           [4, 5],
           [6, 7]])



When looping over N-dimensional arrays, last axis is looped fastest, first is looped slowest


```python
np.ones((4, 3, 2))
```




    array([[[1., 1.],
            [1., 1.],
            [1., 1.]],
    
           [[1., 1.],
            [1., 1.],
            [1., 1.]],
    
           [[1., 1.],
            [1., 1.],
            [1., 1.]],
    
           [[1., 1.],
            [1., 1.],
            [1., 1.]]])



several ways to initialize values


```python
np.ones(3)
```




    array([1., 1., 1.])




```python
np.zeros(3)
```




    array([0., 0., 0.])




```python
rng = np.random.default_rng(0)
rng.random(3)
```




    array([0.63696169, 0.26978671, 0.04097352])



initializing 2d arrays with a tuple


```python
np.ones((3, 2))
```




    array([[1., 1.],
           [1., 1.],
           [1., 1.]])




```python
np.zeros((3, 2))
```




    array([[0., 0.],
           [0., 0.],
           [0., 0.]])




```python
rng.random((3, 2))
```




    array([[0.01652764, 0.81327024],
           [0.91275558, 0.60663578],
           [0.72949656, 0.54362499]])



## Generating random numbers

Use generator.integers to get random ints<br> 
low (inclusive), high (exclusive), use endpoint=True for inclusive high

2x4 array of random ints between 0 and 4:


```python
rng.integers(5, size=(2, 4))
```




    array([[2, 4, 1, 4],
           [3, 0, 1, 4]])



## Get unique items and counts


```python
a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])

np.unique(a)
```




    array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])



get indices of unique values(first index position of unique values)


```python
values, idx = np.unique(a, return_index=True)
print("values: ", values)
print("indices: ", idx)
```

    values:  [11 12 13 14 15 16 17 18 19 20]
    indices:  [ 0  2  3  4  5  6  7 12 13 14]


frequency count


```python
values, counts = np.unique(a, return_counts=True)
print("values: ", values)
print("counts: ", counts)
```

    values:  [11 12 13 14 15 16 17 18 19 20]
    counts:  [3 2 2 2 1 1 1 1 1 1]


Also works with 2d arrays

## Transposing and reshaping a matrix


```python
data.shape
```




    (3, 2)




```python
data.reshape(2, 3)
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
data.reshape(3, 2)
```




    array([[1, 2],
           [3, 4],
           [5, 6]])




```python
arr = np.arange(6).reshape((2, 3))
arr
```




    array([[0, 1, 2],
           [3, 4, 5]])




```python
arr.transpose()
```




    array([[0, 3],
           [1, 4],
           [2, 5]])




```python
arr.T
```




    array([[0, 3],
           [1, 4],
           [2, 5]])



## How to reverse an array


```python
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
np.flip(arr)
```




    array([8, 7, 6, 5, 4, 3, 2, 1])




```python
arr_2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
arr_2d
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
np.flip(arr_2d, axis=0)
```




    array([[ 9, 10, 11, 12],
           [ 5,  6,  7,  8],
           [ 1,  2,  3,  4]])




```python
np.flip(arr_2d, axis=1)
```




    array([[ 4,  3,  2,  1],
           [ 8,  7,  6,  5],
           [12, 11, 10,  9]])



reverse contents of only 1 row


```python
arr_2d[1] = np.flip(arr_2d[1])
arr_2d
```




    array([[ 1,  2,  3,  4],
           [ 8,  7,  6,  5],
           [ 9, 10, 11, 12]])




```python
arr_2d[:,1] = np.flip(arr_2d[:, 1])
arr_2d
```




    array([[ 1, 10,  3,  4],
           [ 8,  7,  6,  5],
           [ 9,  2, 11, 12]])



## Reshaping and flattening multidimensional arrays


```python
x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
x.flatten()
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])



flatten does a deepcopy, not a view


```python
a1 = x.flatten()
a1[0] = 99
x
```




    array([[ 1,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
a1
```




    array([99,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])



ravel returns a view


```python
a2 = x.ravel()
a2[0] = 99
x
```




    array([[99,  2,  3,  4],
           [ 5,  6,  7,  8],
           [ 9, 10, 11, 12]])




```python
a2
```




    array([99,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])



## Access docstring for more information


```python
help(max)
```

    Help on built-in function max in module builtins:
    
    max(...)
        max(iterable, *[, default=obj, key=func]) -> value
        max(arg1, arg2, *args, *[, key=func]) -> value
        
        With a single iterable argument, return its biggest item. The
        default keyword-only argument specifies an object to return if
        the provided iterable is empty.
        With two or more arguments, return the largest argument.
    



```python
a = np.array([1, 2, 3, 4, 5, 6])
```


```python
a?
```

## Working with mathematical formulas

it's super easy... that's about it

## How to save and load NumPy objects


```python
a = np.array([1, 2, 3, 4, 5, 6])
np.save('a', a)
```
