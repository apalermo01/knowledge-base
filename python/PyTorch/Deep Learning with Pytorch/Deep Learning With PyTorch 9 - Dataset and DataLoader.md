# Deep Learning With PyTorch 9 - Dataset and DataLoader

Better way for large datasets: divide model into batches. 

- epoch = 1 forward and backward pass of ALL training samples
- batch_size = number of training samples in one forward and backward pass
- number of iterations = number of passes, each pass using [batch_size] number of samples


```python
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
```


```python
class WineDataset(Dataset):
    
    def __init__(self):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # makes this into the shape the pytorch likes
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        # dataset[index]
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(features)
print(labels)
```

    tensor([1.4230e+01, 1.7100e+00, 2.4300e+00, 1.5600e+01, 1.2700e+02, 2.8000e+00,
            3.0600e+00, 2.8000e-01, 2.2900e+00, 5.6400e+00, 1.0400e+00, 3.9200e+00,
            1.0650e+03])
    tensor([1.])



```python
dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
```

num_workers uses multiple subprocesses to load the data


```python
datatiter = iter(dataloader)
data = datatiter.next()
features, labels = data
print(features)
print(labels)
```

    tensor([[1.3580e+01, 2.5800e+00, 2.6900e+00, 2.4500e+01, 1.0500e+02, 1.5500e+00,
             8.4000e-01, 3.9000e-01, 1.5400e+00, 8.6600e+00, 7.4000e-01, 1.8000e+00,
             7.5000e+02],
            [1.1820e+01, 1.4700e+00, 1.9900e+00, 2.0800e+01, 8.6000e+01, 1.9800e+00,
             1.6000e+00, 3.0000e-01, 1.5300e+00, 1.9500e+00, 9.5000e-01, 3.3300e+00,
             4.9500e+02],
            [1.3170e+01, 2.5900e+00, 2.3700e+00, 2.0000e+01, 1.2000e+02, 1.6500e+00,
             6.8000e-01, 5.3000e-01, 1.4600e+00, 9.3000e+00, 6.0000e-01, 1.6200e+00,
             8.4000e+02],
            [1.3480e+01, 1.8100e+00, 2.4100e+00, 2.0500e+01, 1.0000e+02, 2.7000e+00,
             2.9800e+00, 2.6000e-01, 1.8600e+00, 5.1000e+00, 1.0400e+00, 3.4700e+00,
             9.2000e+02]])
    tensor([[3.],
            [2.],
            [3.],
            [1.]])


Training loop


```python
num_epochs = 2

total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)
print(total_samples, n_iterations)
```

    178 45



```python
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        
        if (i+1) % 5 == 0: 
            print(f"epoch: {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}")
```

    epoch: 1/2, step 5/45, inputs torch.Size([4, 13])
    epoch: 1/2, step 10/45, inputs torch.Size([4, 13])
    epoch: 1/2, step 15/45, inputs torch.Size([4, 13])
    epoch: 1/2, step 20/45, inputs torch.Size([4, 13])
    epoch: 1/2, step 25/45, inputs torch.Size([4, 13])
    epoch: 1/2, step 30/45, inputs torch.Size([4, 13])
    epoch: 1/2, step 35/45, inputs torch.Size([4, 13])
    epoch: 1/2, step 40/45, inputs torch.Size([4, 13])
    epoch: 1/2, step 45/45, inputs torch.Size([2, 13])
    epoch: 2/2, step 5/45, inputs torch.Size([4, 13])
    epoch: 2/2, step 10/45, inputs torch.Size([4, 13])
    epoch: 2/2, step 15/45, inputs torch.Size([4, 13])
    epoch: 2/2, step 20/45, inputs torch.Size([4, 13])
    epoch: 2/2, step 25/45, inputs torch.Size([4, 13])
    epoch: 2/2, step 30/45, inputs torch.Size([4, 13])
    epoch: 2/2, step 35/45, inputs torch.Size([4, 13])
    epoch: 2/2, step 40/45, inputs torch.Size([4, 13])
    epoch: 2/2, step 45/45, inputs torch.Size([2, 13])

