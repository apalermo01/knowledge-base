```python
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
```


```python
# train, val split
train_data = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)
```


```python
# define model
model = nn.Sequential(
  nn.Linear(28*28, 64),
  nn.ReLU(),
  nn.Linear(64, 64), 
  nn.ReLU(),
  nn.Dropout(0.1),
  nn.Linear(64, 10)
)
```


```python
# define a more flexible model
class ResNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(28*28, 64)
    self.l2 = nn.Linear(64, 64)
    self.l3 = nn.Linear(64, 10)
    self.do = nn.Dropout(0.1)
    
  def forward(self, x):
    h1 = nn.functional.relu(self.l1(x))
    h2 = nn.functional.relu(self.l2(h1))
    do = self.do(h2+h1)
    logits = self.l3(do)
    return logits
model = ResNet()
```


```python
# Define optimizer
params = model.parameters()
optimiser = optim.SGD(params, lr=1e-2)
```


```python
# Define loss
loss = nn.CrossEntropyLoss()
```


```python
# training and validation loops
nb_epochs = 5
for epoch in range(nb_epochs):
  losses = list()
  accuracies = list()
  model.train()
  for batch in train_loader:
    x, y = batch
    
    # x: b x 1 x 28 x 28
    b = x.size(0)
    x = x.view(b, -1)
    
    # 1 forward
    l = model(x) # l: logits
    
    # 2 compute objective function
    J = loss(l, y)
    
    # 3 clearning the gradients
    model.zero_grad()
    
    # accumulate the partial derivatives of J wrt params
    J.backward()
    
    # 5 step in the opposite direction of the gradient
    optimiser.step()
    
    losses.append(J.item())
    accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
    
  print(f'Epoch {epoch+1}', end=', ')
  print(f'training loss: {torch.tensor(losses).mean():.2f}', end=', ')
  print(f'training accuracy: {torch.tensor(accuracies).mean():.2f}')
  
  losses = list()
  accuracies = list()
  model.eval()
  for batch in val_loader:
    x, y = batch
    
    # x: b x 1 x 28 x 28
    b = x.size(0)
    x = x.view(b, -1)
    
    # 1 forward
    with torch.no_grad():
      l = model(x) # l: logits
    
    # 2 compute objective function
    J = loss(l, y)
    
    losses.append(J.item())
    accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

  print(f'Epoch {epoch+1}', end=', ')
  print(f'validation loss: {torch.tensor(losses).mean():.2f}', end=', ')
  print(f'validation accuracy: {torch.tensor(accuracies).mean():.2f}')
```

    Epoch 1, training loss: 0.81, training accuracy: 0.79
    Epoch 1, validation loss: 0.40, validation accuracy: 0.89
    Epoch 2, training loss: 0.37, training accuracy: 0.89
    Epoch 2, validation loss: 0.32, validation accuracy: 0.91
    Epoch 3, training loss: 0.30, training accuracy: 0.91
    Epoch 3, validation loss: 0.28, validation accuracy: 0.92
    Epoch 4, training loss: 0.26, training accuracy: 0.92
    Epoch 4, validation loss: 0.25, validation accuracy: 0.93
    Epoch 5, training loss: 0.23, training accuracy: 0.93
    Epoch 5, validation loss: 0.22, validation accuracy: 0.94

