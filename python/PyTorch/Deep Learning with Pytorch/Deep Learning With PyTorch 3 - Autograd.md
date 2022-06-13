# Deep Learning With PyTorch 3 - Autograd


```python
import torch
```


```python
x = torch.randn(3)
x
```




    tensor([ 0.1682, -0.3163, -2.5467])



If we want to calculate the gradients later:


```python
x = torch.randn(3, requires_grad=True)
x
```




    tensor([ 0.3174, -1.1878, -1.8222], requires_grad=True)



This will automatically create a computational graph


```python
y = x + 2
```

![image.png](attachment:image.png)

Back propagation: calculate gradients. For now, just focusing on how to use it. 

Forward pass: calculate output

Since requires_grad is true, pytorch will automatically create a function to do back propagation

y has an attribute grad_fn


```python
y
```




    tensor([2.3174, 0.8122, 0.1778], grad_fn=<AddBackward0>)




```python
y.grad_fn
```




    <AddBackward0 at 0x242047318b0>



This AddBackward will apply backpropagation through this operation


```python
z = y*y*2
z
```




    tensor([10.7404,  1.3193,  0.0632], grad_fn=<MulBackward0>)




```python
z = z.mean()
z
```




    tensor(4.0410, grad_fn=<MeanBackward0>)



This will calculate the gradient of z with respect to x


```python
z.backward() #
```

This stores the gradients to x.grad


```python
print(x.grad)
```

    tensor([3.0898, 1.0829, 0.2370])


What if we don't specify requires_grad = True? 


```python
x = torch.randn(3)
y = x + 2
y.backward()
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-15-0c6c26143d3d> in <module>
          1 x = torch.randn(3)
          2 y = x + 2
    ----> 3 y.backward()
    

    ~\anaconda3\envs\Armada_AV\lib\site-packages\torch\tensor.py in backward(self, gradient, retain_graph, create_graph)
        183                 products. Defaults to ``False``.
        184         """
    --> 185         torch.autograd.backward(self, gradient, retain_graph, create_graph)
        186 
        187     def register_hook(self, hook):


    ~\anaconda3\envs\Armada_AV\lib\site-packages\torch\autograd\__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables)
        123         retain_graph = create_graph
        124 
    --> 125     Variable._execution_engine.run_backward(
        126         tensors, grad_tensors, retain_graph, create_graph,
        127         allow_unreachable=True)  # allow_unreachable flag


    RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn


Mathematically, this creates a vector jacobian matrix of a bunch of partial derivates -> chain rule on steroids

![image.png](attachment:image.png)


```python
x = torch.randn(3, requires_grad=True)
y = x + 2
z = y*y*2
print(z)
```

    tensor([ 0.1007,  0.1431, 11.8733], grad_fn=<MulBackward0>)


If z is not a scalar, we need an argument for backward()<br> 
Usually, we wind up creating a scalar value (this is probably the value of the error funciton)<br> 


```python
# create a vector of the same size
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)
```


```python
x.grad
```




    tensor([0.0898, 1.0700, 0.0097])



Sometimes during training, weight updates should not be part of gradient operation. We need to prevent pytorch from tracking gradients. 

1) call x.requires_grad_(false)<br>
2) call x.detach -> creates new tensor<br>
3) wrap in: with torch.no_grad():<br>

trailing underscore = modifies variable in place


```python
x.requires_grad_(False)
x
```




    tensor([-1.7756, -1.7325,  0.4365])




```python
x = torch.randn(3, requires_grad=True)
y = x.detach()
print(x)
y
```

    tensor([-0.1922, -0.5988,  1.0309], requires_grad=True)





    tensor([-0.1922, -0.5988,  1.0309])




```python
with torch.no_grad():
    y = x + 2
    print(y)
```

    tensor([1.8078, 1.4012, 3.0309])


We must be very careful since gradients can accumulate

Do some dummy training for illustration:


```python
weights = torch.ones(4, requires_grad=True)
for epoch in range(100):
    
    model_output = (weights*3).sum()
    
    model_output.backward()
    
    if epoch % 10 == 0:
        print(weights.grad)
```

    tensor([3., 3., 3., 3.])
    tensor([33., 33., 33., 33.])
    tensor([63., 63., 63., 63.])
    tensor([93., 93., 93., 93.])
    tensor([123., 123., 123., 123.])
    tensor([153., 153., 153., 153.])
    tensor([183., 183., 183., 183.])
    tensor([213., 213., 213., 213.])
    tensor([243., 243., 243., 243.])
    tensor([273., 273., 273., 273.])


Before we do the next step, we need to empty the gradients


```python
weights = torch.ones(4, requires_grad=True)
for epoch in range(100):
    
    model_output = (weights*3).sum()
    
    model_output.backward()
    
    if epoch % 10 == 0:
        print(weights.grad)
    weights.grad.zero_()
```

    tensor([3., 3., 3., 3.])
    tensor([3., 3., 3., 3.])
    tensor([3., 3., 3., 3.])
    tensor([3., 3., 3., 3.])
    tensor([3., 3., 3., 3.])
    tensor([3., 3., 3., 3.])
    tensor([3., 3., 3., 3.])
    tensor([3., 3., 3., 3.])
    tensor([3., 3., 3., 3.])
    tensor([3., 3., 3., 3.])


Now everything is correct!

Later: we will work with builtin optimizers. before the next step, we need to run zero_grad() on the optimizer. 
