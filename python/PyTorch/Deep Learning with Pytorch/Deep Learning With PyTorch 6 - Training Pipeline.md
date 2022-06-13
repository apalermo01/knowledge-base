# Deep Learning With PyTorch 6 - Training Pipeline

replace prediction, loss, and parameters with pytorch. 

**Original**


```python
import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# init weight
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w*x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# training
learning_rate = 0.1
n_iters = 20

for epoch in range(n_iters):
    
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw

    
    # update weights
        
    with torch.no_grad():
        w -= w.grad * learning_rate
    w.grad.zero_()
    
    if epoch%1 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")
print(f"Prediction after training: f(5) = {forward(5):.3f}")
```

    Prediction before training: f(5) = 0.000
    epoch 1: w = 3.000, loss = 30.00000000
    epoch 2: w = 1.500, loss = 7.50000000
    epoch 3: w = 2.250, loss = 1.87500000
    epoch 4: w = 1.875, loss = 0.46875000
    epoch 5: w = 2.062, loss = 0.11718750
    epoch 6: w = 1.969, loss = 0.02929688
    epoch 7: w = 2.016, loss = 0.00732422
    epoch 8: w = 1.992, loss = 0.00183105
    epoch 9: w = 2.004, loss = 0.00045776
    epoch 10: w = 1.998, loss = 0.00011444
    epoch 11: w = 2.001, loss = 0.00002861
    epoch 12: w = 2.000, loss = 0.00000715
    epoch 13: w = 2.000, loss = 0.00000179
    epoch 14: w = 2.000, loss = 0.00000045
    epoch 15: w = 2.000, loss = 0.00000011
    epoch 16: w = 2.000, loss = 0.00000003
    epoch 17: w = 2.000, loss = 0.00000001
    epoch 18: w = 2.000, loss = 0.00000000
    epoch 19: w = 2.000, loss = 0.00000000
    epoch 20: w = 2.000, loss = 0.00000000
    Prediction after training: f(5) = 10.000


## General training pipeline in pytorch

1) Design model (input size, outptu size, forward pass)<br> 
2) Construct loss and optimizer<br> 
3) Training loop<br>

- forward pass: compute the prediction
- backward pass: gradients
- Update weights
- iterate until we're done!

Replace loss and optimization


```python
import torch
import torch.nn as nn

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

# init weight
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

# model prediction
def forward(x):
    return w*x

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iters):
    
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw

    
    # update weights
    optimizer.step()
    
    optimizer.zero_grad()
    
    if epoch%10 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")
print(f"Prediction after training: f(5) = {forward(5):.3f}")
```

    Prediction before training: f(5) = 0.000
    epoch 1: w = 0.300, loss = 30.00000000
    epoch 11: w = 1.665, loss = 1.16278565
    epoch 21: w = 1.934, loss = 0.04506890
    epoch 31: w = 1.987, loss = 0.00174685
    epoch 41: w = 1.997, loss = 0.00006770
    epoch 51: w = 1.999, loss = 0.00000262
    epoch 61: w = 2.000, loss = 0.00000010
    epoch 71: w = 2.000, loss = 0.00000000
    epoch 81: w = 2.000, loss = 0.00000000
    epoch 91: w = 2.000, loss = 0.00000000
    Prediction after training: f(5) = 10.000


Replace forward with pytorch model<br>
Make x an y 2d array<br> 
num rows = num samples<br>


```python
import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")

# training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    
    # prediction = forward pass
    y_pred = model(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw

    
    # update weights
    optimizer.step()
    
    optimizer.zero_grad()
    
    if epoch%10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0].item():.3f}, loss = {l:.8f}")
print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")
```

    Prediction before training: f(5) = 1.893
    epoch 1: w = 0.431, loss = 17.12932777
    epoch 11: w = 1.384, loss = 0.71142823
    epoch 21: w = 1.548, loss = 0.27104411
    epoch 31: w = 1.585, loss = 0.24494536
    epoch 41: w = 1.601, loss = 0.23042139
    epoch 51: w = 1.613, loss = 0.21700262
    epoch 61: w = 1.625, loss = 0.20437184
    epoch 71: w = 1.636, loss = 0.19247638
    epoch 81: w = 1.647, loss = 0.18127325
    epoch 91: w = 1.657, loss = 0.17072232
    Prediction after training: f(5) = 9.313


Let's say we'd like to make a custom model


```python
import torch
import torch.nn as nn

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)
X_test = torch.tensor([5], dtype=torch.float32)
n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        
        # this super will call the super's constructor module
        super(LinearRegression, self).__init__()
        
        # define layers
        self.lin = nn.Linear(input_dim, output_dim)
        
    # implement forward pass
    def forward(self, x):
        return self.lin(x)
    
model = LinearRegression(input_size, output_size)


print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")

# training
learning_rate = 0.01
n_iters = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    
    # prediction = forward pass
    y_pred = model(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients = backward pass
    l.backward() # dl/dw

    
    # update weights
    optimizer.step()
    
    optimizer.zero_grad()
    
    if epoch%10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0].item():.3f}, loss = {l:.8f}")
print(f"Prediction after training: f(5) = {model(X_test).item():.3f}")
```

    Prediction before training: f(5) = 2.731
    epoch 1: w = 0.827, loss = 16.70854568
    epoch 11: w = 1.765, loss = 0.43668324
    epoch 21: w = 1.917, loss = 0.01543556
    epoch 31: w = 1.943, loss = 0.00429605
    epoch 41: w = 1.948, loss = 0.00378105
    epoch 51: w = 1.950, loss = 0.00355410
    epoch 61: w = 1.952, loss = 0.00334706
    epoch 71: w = 1.953, loss = 0.00315224
    epoch 81: w = 1.955, loss = 0.00296876
    epoch 91: w = 1.956, loss = 0.00279596
    Prediction after training: f(5) = 9.912



```python

```
