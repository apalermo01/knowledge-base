# Deep Learning With PyTorch 2 - Tensor Basics
https://www.youtube.com/watch?v=c36lUUr864M

Everything is based on tensor operations. 
- Can have any number of dimensions

Create an empty tensor of different sizes


```python
import torch

x = torch.empty(1)
print(x)
```

    tensor([0.])



```python
x = torch.empty(3)
print(x)
```

    tensor([1.7753e+28, 1.3458e-14, 7.8448e+17])



```python
x = torch.empty(2, 3)
print(x)
```

    tensor([[1.3590e+22, 4.1291e-05, 4.1577e+21],
            [1.6926e+22, 4.0973e-11, 4.2330e+21]])



```python
x = torch.empty(2, 2, 2, 2)
print(x)
```

    tensor([[[[9.2755e-39, 1.0561e-38],
              [8.9082e-39, 9.2755e-39]],
    
             [[9.0919e-39, 4.2246e-39],
              [1.0286e-38, 1.0653e-38]]],
    
    
            [[[1.0194e-38, 8.4490e-39],
              [1.0469e-38, 9.3674e-39]],
    
             [[9.9184e-39, 8.7245e-39],
              [9.2755e-39, 8.9082e-39]]]])


**Other methods for initializing tensors**
torch.random()
torch.zeros()
torch.ones()

datatypes


```python
x.dtype
```




    torch.float32



float32 by default.

Can specify datatype during initialization


```python
x = torch.empty(2, dtype=torch.int32)
x
```




    tensor([-1797939728,       32760], dtype=torch.int32)




```python
x.size()
```




    torch.Size([2])



can construct tensors from other datatypes


```python
x = torch.tensor([2.5, 0.1])
x
```




    tensor([2.5000, 0.1000])



**Tensor operations**


```python
x = torch.rand(2, 2)
y = torch.rand(2, 2)
print(x)
print(y)
```

    tensor([[0.7913, 0.0284],
            [0.8455, 0.0326]])
    tensor([[0.9528, 0.5208],
            [0.9833, 0.7684]])



```python
z = x + y
z
```




    tensor([[1.7441, 0.5493],
            [1.8288, 0.8010]])




```python
z = torch.add(x, y)
z
```




    tensor([[1.7441, 0.5493],
            [1.8288, 0.8010]])



Addition in place, modifies y


```python
y.add_(x)
print(y)
```

    tensor([[1.7441, 0.5493],
            [1.8288, 0.8010]])


Every function with _<name> does an inplace operation - modifies the variable that is is applied on


```python
z = x - y
z = torch.sub(x, y)
z
```




    tensor([[-0.9528, -0.5208],
            [-0.9833, -0.7684]])




```python
z = torch.mul(x, y)
print(z)
```

    tensor([[1.3802, 0.0156],
            [1.5463, 0.0261]])


Same trend with all the basic operations

**Slicing operations**
basically the same as numpy arrays


```python
x = torch.rand(5, 3)
x
```




    tensor([[0.9206, 0.3275, 0.8231],
            [0.6739, 0.1803, 0.9109],
            [0.0688, 0.8515, 0.6428],
            [0.4119, 0.8909, 0.2463],
            [0.9637, 0.0502, 0.3632]])




```python
x[1]
```




    tensor([0.6739, 0.1803, 0.9109])




```python
x[:, 1]
```




    tensor([0.3275, 0.1803, 0.8515, 0.8909, 0.0502])




```python
x[1, 0]
```




    tensor(0.6739)



.item() returns the actual value, but only works on single datapoints


```python
x[1, 0].item()
```




    0.6738866567611694



**re-shaping**


```python
x = torch.rand(4, 4)
print(x) 
y = x.view(16)
print(y)
```

    tensor([[0.2867, 0.3498, 0.3490, 0.0895],
            [0.5149, 0.0365, 0.6158, 0.1435],
            [0.8038, 0.7098, 0.6676, 0.2491],
            [0.2739, 0.9186, 0.6970, 0.4815]])
    tensor([0.2867, 0.3498, 0.3490, 0.0895, 0.5149, 0.0365, 0.6158, 0.1435, 0.8038,
            0.7098, 0.6676, 0.2491, 0.2739, 0.9186, 0.6970, 0.4815])



```python
x.view(-1, 8)
```




    tensor([[0.2867, 0.3498, 0.3490, 0.0895, 0.5149, 0.0365, 0.6158, 0.1435],
            [0.8038, 0.7098, 0.6676, 0.2491, 0.2739, 0.9186, 0.6970, 0.4815]])




```python
print(x.view(-1, 8).size())
```

    torch.Size([2, 8])


**Converting numpy to torch**


```python
import numpy as np
```


```python
a = torch.ones(5)
a
```




    tensor([1., 1., 1., 1., 1.])




```python
b = a.numpy()
b
```




    array([1., 1., 1., 1., 1.], dtype=float32)



If we modify b or a inplace: 


```python
a.add_(1)
a
```




    tensor([2., 2., 2., 2., 2.])




```python
b
```




    array([2., 2., 2., 2., 2.], dtype=float32)



The tensor and numpy array occupy the same memory location. I think this works differently on a GPU


```python
a = np.ones(5)
a
```




    array([1., 1., 1., 1., 1.])




```python
b = torch.from_numpy(a)
b
```




    tensor([1., 1., 1., 1., 1.], dtype=torch.float64)



Again, if we modify a in place, we also modify b

But we can create a tensor right on the GPU like this: 


```python
if torch.cuda.is_available():
    print("Cuda is available")
    device = torch.device("cuda")
    x = torch.ones(5, device=device) # creates a tensor on the GPU
    y = torch.ones(5)
    # can move a variable to gpu
    y = y.to(device)
    
    # calling: 
    y.to_numpy()
    
    # will throw an error becuase numpy can't handle GPU operations
    
    z = x + y
    # we can fix this by doing
    z.to("cpu")
else: 
    print("Cuda not available")
```

    Cuda not available


We can tell pytorch ahead of time when we will be required to calculate the gradient of a tensor


```python
x = torch.ones(5, requires_grad=True)
x
```




    tensor([1., 1., 1., 1., 1.], requires_grad=True)


