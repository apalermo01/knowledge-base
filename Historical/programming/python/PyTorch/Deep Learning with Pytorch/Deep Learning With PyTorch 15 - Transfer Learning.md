# Deep Learning With PyTorch 15 - Transfer Learning

A model developed for a first task is used to work on a second text. 
Ex: take a model used to classify birds and cats and re-purpose the model to classify bees and dogs. 

Take a pre-trained model and modify the last few layers to classify different things.

![image.png](attachment:image.png)


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


```


```python
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}
```


```python
data_dir = 'data/'
sets = ['train', 'val']
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}
```


    ---------------------------------------------------------------------------

    FileNotFoundError                         Traceback (most recent call last)

    <ipython-input-7-df83f026fb48> in <module>
          1 data_dir = 'data/'
          2 sets = ['train', 'val']
    ----> 3 image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
          4                   for x in ['train', 'val']}
          5 


    <ipython-input-7-df83f026fb48> in <dictcomp>(.0)
          1 data_dir = 'data/'
          2 sets = ['train', 'val']
    ----> 3 image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
          4                   for x in ['train', 'val']}
          5 


    ~\anaconda3\envs\Armada_AV\lib\site-packages\torchvision\datasets\folder.py in __init__(self, root, transform, target_transform, loader, is_valid_file)
        203     def __init__(self, root, transform=None, target_transform=None,
        204                  loader=default_loader, is_valid_file=None):
    --> 205         super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
        206                                           transform=transform,
        207                                           target_transform=target_transform,


    ~\anaconda3\envs\Armada_AV\lib\site-packages\torchvision\datasets\folder.py in __init__(self, root, loader, extensions, transform, target_transform, is_valid_file)
         92         super(DatasetFolder, self).__init__(root, transform=transform,
         93                                             target_transform=target_transform)
    ---> 94         classes, class_to_idx = self._find_classes(self.root)
         95         samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
         96         if len(samples) == 0:


    ~\anaconda3\envs\Armada_AV\lib\site-packages\torchvision\datasets\folder.py in _find_classes(self, dir)
        121             No class is a subdirectory of another.
        122         """
    --> 123         classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        124         classes.sort()
        125         class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}


    FileNotFoundError: [WinError 3] The system cannot find the path specified: 'data/train'


Load a pretrained dataset


```python
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
num_ftrs = model.fc.in_features # number of features in last layer

# create a new layer and put it on the last layer
model.fc = nn.Linear(num_ftrs, 2)
model.to('cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

```

Scheduler


```python
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# every 7 epochs, our learning rate is multiplied by 0.1

model = train_model(model, criterion, optimizer, scheduler, num_epochs=20)
```
