# Deep Learning With PyTorch 4 - Backpropagation

![image.png](attachment:image.png)

**Computational graph**

node = calculation
can calculate local gradients that we use later

![image-3.png](attachment:image-3.png)

1) Forward pass - compute loss<br> 
2) compute local gradients<br> 
3) backward pass: compute dLoss/dWeights

**Linear Regression Example**

![image.png](attachment:image.png)

![image.png](attachment:image.png)


```python
import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
loss = (y_hat - y)**2

print(loss)
```

    tensor(1., grad_fn=<PowBackward0>)



```python
# backward pass

# gradient computation
loss.backward()

print(w.grad)
```

    tensor(-2.)



```python
# Update weigths

# Next forward and backward pass
```
