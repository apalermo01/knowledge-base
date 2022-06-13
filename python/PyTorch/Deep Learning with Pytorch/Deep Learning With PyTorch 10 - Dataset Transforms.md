# Deep Learning With PyTorch 10 - Dataset Transforms


```python
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
```


```python
class WineDataset(Dataset):
    
    def __init__(self, transform):
        # data loading
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.transform = transform
        self.n_samples = xy.shape[0]
        
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples
```

Create some custom transform classes


```python
class ToTensor:
    
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
```


```python
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features, labels)
print(type(features), type(labels))
```

    tensor([1.4230e+01, 1.7100e+00, 2.4300e+00, 1.5600e+01, 1.2700e+02, 2.8000e+00,
            3.0600e+00, 2.8000e-01, 2.2900e+00, 5.6400e+00, 1.0400e+00, 3.9200e+00,
            1.0650e+03]) tensor([1.])
    <class 'torch.Tensor'> <class 'torch.Tensor'>



```python
class MulTransform:
    
    def __init__(self, factor):
        self.factor = factor
    
    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets
```


```python
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features, labels)
print(type(features), type(labels))
```

    tensor([5.6920e+01, 6.8400e+00, 9.7200e+00, 6.2400e+01, 5.0800e+02, 1.1200e+01,
            1.2240e+01, 1.1200e+00, 9.1600e+00, 2.2560e+01, 4.1600e+00, 1.5680e+01,
            4.2600e+03]) tensor([1.])
    <class 'torch.Tensor'> <class 'torch.Tensor'>

