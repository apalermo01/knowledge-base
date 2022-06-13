# Deep Learning With PyTorch 7 - Linear Regression

## General training pipeline in pytorch

1) Design model (input size, outptu size, forward pass)<br> 
2) Construct loss and optimizer<br> 
3) Training loop<br>

- forward pass: compute the prediction
- backward pass: gradients
- Update weights
- iterate until we're done!


```python
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
```


```python
# 0) prepare_data
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0], 1) # reshape y

n_samples, n_features = X.shape

# 1) model
model = nn.Linear(n_features, 1)

# 2) loss and optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) training loop
num_epochs = 1000
for epoch in range(num_epochs):
    
    # forward pass and loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # backward pass
    loss.backward()
    
    # update
    optimizer.step()
    optimizer.zero_grad()
    
    if (epoch+1) % 100 == 0: 
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

# plot
predicted = model(X).detach().numpy()
plt.plot(X_numpy, y_numpy, 'ro')
plt.plot(X_numpy, predicted, 'b')
```

    epoch: 100, loss = 567.4867
    epoch: 200, loss = 342.7598
    epoch: 300, loss = 333.0179
    epoch: 400, loss = 332.5876
    epoch: 500, loss = 332.5685
    epoch: 600, loss = 332.5676
    epoch: 700, loss = 332.5675
    epoch: 800, loss = 332.5675
    epoch: 900, loss = 332.5676
    epoch: 1000, loss = 332.5676





    [<matplotlib.lines.Line2D at 0x2c496e54b20>]




    
![png](Deep%20Learning%20With%20PyTorch%207%20-%20Linear%20Regression_files/Deep%20Learning%20With%20PyTorch%207%20-%20Linear%20Regression_3_2.png)
    

