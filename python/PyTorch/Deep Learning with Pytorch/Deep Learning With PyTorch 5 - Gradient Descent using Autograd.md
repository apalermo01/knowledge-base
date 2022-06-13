# Deep Learning With PyTorch 5 - Gradient Descent using Autograd


Implement linear regression manually, and then replace each step with pytorch's implementation


```python
import numpy as np

X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

# init weight
w = 0

# model prediction
def forward(x):
    return w*x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted - y) ** 2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x (w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    
    # prediction = forward pass
    y_pred = forward(X)
    
    # loss
    l = loss(Y, y_pred)
    
    # gradients
    dw = gradient(X, Y, y_pred)
    
    # update weights
    w -= dw * learning_rate
    
    if epoch%10 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")
print(f"Prediction after training: f(5) = {forward(5):.3f}")
```

    Prediction before training: f(5) = 0.000
    epoch 1: w = 1.200, loss = 30.00000000
    epoch 11: w = 2.000, loss = 0.00000033
    epoch 21: w = 2.000, loss = 0.00000000
    epoch 31: w = 2.000, loss = 0.00000000
    epoch 41: w = 2.000, loss = 0.00000000
    epoch 51: w = 2.000, loss = 0.00000000
    epoch 61: w = 2.000, loss = 0.00000000
    epoch 71: w = 2.000, loss = 0.00000000
    epoch 81: w = 2.000, loss = 0.00000000
    epoch 91: w = 2.000, loss = 0.00000000
    Prediction after training: f(5) = 10.000


Now replace the gradient calculation


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


This requires more iterations before because pytorch's gradient descent is not a numerically accurate as the manual implementation
