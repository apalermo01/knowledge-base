# The N-dimensional array (ndarray)

- ndarray = usually fixed-sized multidimensional array of items with same types and size. 
- shape = tuple of integers specifying the sizes of each dimension


```python
import numpy as np

x = np.array([[5, 2, 3], [8, 2, 4]], np.int32)
type(x)
```




    numpy.ndarray




```python
x
```




    array([[5, 2, 3],
           [8, 2, 4]])




```python
x.shape
```




    (2, 3)




```python
x.dtype
```




    dtype('int32')



Some indexing


```python
# second row, 3rd column
x[1, 2]
```




    4



Numpy arrays can be sliced much like DataFrames


```python
# second column, every row
y = x[:, 1]
y
```




    array([2, 2])




```python
y[0]
```




    2



Here, y is a **view** of x, if we change x, we change y


```python
x[0, 1] = 11
x
```




    array([[ 5, 11,  3],
           [ 8,  2,  4]])




```python
y[0]
```




    11



If we change y, we change x


```python
y[0] = 13
x
```




    array([[ 5, 13,  3],
           [ 8,  2,  4]])



# useful ndarray attributes

- x.shape = array shape
- x.ndim = number of dimensions
- x.size = number of element
- x.T = transposed array
- x.real = real part of array
- x.imag = imaginary part of array
- x.flat = 1D iterator


```python
print("x =", x)
print("\nx.shape")
print(x.shape)
print("\nx.ndim")
print(x.ndim)
print("\nx.size")
print(x.size)
print("\nx.T")
print(x.T)
print("\nx.real")
print(x.real)
print("\nx.img")
print(x.imag)
print("\nx.flat")
print(list(x.flat))
```

    x= [[ 5 13  3]
     [ 8  2  4]]
    
    x.shape
    (2, 3)
    
    x.ndim
    2
    
    x.size
    6
    
    x.T
    [[ 5  8]
     [13  2]
     [ 3  4]]
    
    x.real
    [[ 5 13  3]
     [ 8  2  4]]
    
    x.img
    [[0 0 0]
     [0 0 0]]
    
    x.flat
    [5, 13, 3, 8, 2, 4]

