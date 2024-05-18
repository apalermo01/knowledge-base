# Linear algebra on n-dimensional arrays

use singular value decomposition to generate a compressed approximation of an image

https://numpy.org/doc/1.21/user/tutorial-svd.html


```python
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
```


```python
img = misc.face()
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x7fcd4e81bdd0>




    
![png](Tutorial-%20Linear%20algebra%20on%20n-dimensional%20arrays_2_1.png)
    


## Shape, axis, and array properties


```python
img.shape
```




    (768, 1024, 3)




```python
img.ndim
```




    3



access red channel


```python
img[:, :, 0]
```




    array([[121, 138, 153, ..., 119, 131, 139],
           [ 89, 110, 130, ..., 118, 134, 146],
           [ 73,  94, 115, ..., 117, 133, 144],
           ...,
           [ 87,  94, 107, ..., 120, 119, 119],
           [ 85,  95, 112, ..., 121, 120, 120],
           [ 85,  97, 111, ..., 120, 119, 118]], dtype=uint8)



change scale from 0-255 to 0-1


```python
img_array = img / 255
```

check max and min


```python
print(f"max = {img_array.max()}; min = {img_array.min()}")
```

    max = 1.0; min = 0.0


check dtype


```python
img_array.dtype
```




    dtype('float64')



Color channels: 0=red, 1=green, 2=blue

## Operations on an axis

use SVD to rebuild an image with less info than the original one

note: using **linalg** module. Sometimes **scipy.linalg** is faster but only works with 2d arrays. 


```python
from numpy import linalg
```

Given a matrix $A$, we're going to compute

$$
U \Sigma V^T = A
$$

- $U$, $V^T$ are squares, $\Sigma$ is the same size as $A$ and is a diagonal matrix with the singular values of A
- Singular values describe the importance of features

try on one matrix first: get grayscale image using
$$
Y = 0.2126R + 0.7152G + 0.0722B
$$


```python
img_gray = img_array @ [0.2126, 0.7152, 0.0722]
plt.imshow(img_gray, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x7fcd4cb7ec50>




    
![png](Tutorial-%20Linear%20algebra%20on%20n-dimensional%20arrays_19_1.png)
    



```python
img_gray.shape
```




    (768, 1024)



now run svd


```python
U, s, Vt = linalg.svd(img_gray)
```


```python
print(f"U shape = {U.shape}; s shape = {s.shape}; Vt shape = {Vt.shape}")
```

    U shape = (768, 768); s shape = (768,); Vt shape = (1024, 1024)


s and Vt can't be multiplied out of the box, since it's more efficient to store s as a 1d array


```python
s@Vt
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    /tmp/ipykernel_33069/4053168034.py in <module>
    ----> 1 s@Vt
    

    ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 1024 is different from 768)


generate diagonal matrix


```python
Sigma = np.zeros((768, 1024))
for i in range(768):
  Sigma[i,i] = s[i]
```

## Approximation

use `norm` to calculate the quality of the approximation


```python
linalg.norm(img_gray - U@Sigma@Vt)
```




    1.4102987503620689e-12



can also use `allclose`


```python
np.allclose(img_gray, U@Sigma@Vt)
```




    True




```python
plt.plot(s)
```




    [<matplotlib.lines.Line2D at 0x7fcd4c49a750>]




    
![png](Tutorial-%20Linear%20algebra%20on%20n-dimensional%20arrays_32_1.png)
    


Try considering all but the first k values in Sigma as 0, then computing the product (e.g. only reconstruct the image based on the first few most important features)


```python
k = 10
approx = U @ Sigma[:, :k] @ Vt[:k, :]
plt.imshow(approx, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x7fcd4c232750>




    
![png](Tutorial-%20Linear%20algebra%20on%20n-dimensional%20arrays_34_1.png)
    


## Applying to all colors


```python
img_array.shape
```




    (768, 1024, 3)



linalg functions expect (num matrices, M, M). i.e. reshape the image to (3, 768, 1024)


```python
img_array_transposed = np.transpose(img_array, (2, 0, 1))
img_array_transposed.shape
```




    (3, 768, 1024)




```python
U, s, Vt = linalg.svd(img_array_transposed)
print(f"U shape = {U.shape}; s shape = {s.shape}; Vt shape = {Vt.shape}")
```

    U shape = (3, 768, 768); s shape = (3, 768); Vt shape = (3, 1024, 1024)


## Products with n-dimensional arrays

np.dot, np.matmul, and @ work in very different ways for ndarrays (see docs for np.matmul)

buld sigma matrix- want (3, 768, 1024). Use each row in s as a diagonal for each channel


```python
Sigma = np.zeros((3, 768, 1024))
for j in range(3):
  np.fill_diagonal(Sigma[j, :, :], s[j, :])
```


```python
reconstructed = U@Sigma@Vt
```


```python
reconstructed.shape
```




    (3, 768, 1024)




```python
plt.imshow(np.transpose(reconstructed, (1, 2, 0)))
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).





    <matplotlib.image.AxesImage at 0x7fcd4c13d750>




    
![png](Tutorial-%20Linear%20algebra%20on%20n-dimensional%20arrays_44_2.png)
    


do approximation


```python
k = 20
approx_img = U @ Sigma[..., :k] @ Vt[..., :k, :]
approx_img.shape
```




    (3, 768, 1024)




```python
plt.imshow(np.transpose(approx_img, (1, 2, 0)))
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).





    <matplotlib.image.AxesImage at 0x7fcd4c03ad50>




    
![png](Tutorial-%20Linear%20algebra%20on%20n-dimensional%20arrays_47_2.png)
    


## Final words

better algorithms exist (e.g. faster)- but this is the best approximation we can get to when considering the norm of the difference.
