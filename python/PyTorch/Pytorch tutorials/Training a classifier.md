# Training a classifier tutorial

source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

**imports**


```python
import torch
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from tqdm.notebook import tqdm
```

defining some path variables


```python
model_path = "./cifar_net.pth"
root = "/home/alex/datasets/"
```

# 1. Data

3 main domains: images, audio, and text

- images: Pillow, OpenCV
    - for this tutorial, using torchvision
- audio: scipy, librosa
- text: raw python / cython, or NLTK, SpaCy

**Objective: load and normalize CIFAR10 dataset**


```python
# Build the data transformation pipeline. First, convert all images to pytorch tensor,
# then transforms each channel in the image (3-tuple, so 3 channels) to have a 
# mean of 0.5 (first tuple) and std of 0.5 (second tuple)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# hyperparameters
batch_size = 4

# other parameters
num_workers = 8

# load dataset and create dataloaders
trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                       download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                         shuffle=True, num_workers=num_workers)

testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

    Files already downloaded and verified
    Files already downloaded and verified


**Notes**
- best practice is to shuffle the train set but not the test set
- usually, you want to set num_workers = number of cpu cores on your computer

**visualize some training images**


```python
def show_image(img):
    img = img / 2 + 0.5 # unnormalize the image (remember in the transform pipeline we set the images to 
    # have mean 0.5 and std 0.5) TODO: are we sure this is correct? doesn't seem to be consistent
    
    img_numpy = img.numpy()
    
    # img is originally [C, H, W], transpose to [H, W, C]
    plt.imshow(np.transpose(img_numpy, (1, 2, 0)))
    
    plt.show()
```


```python
# get some images
data_iterable = iter(trainloader)
images, labels = data_iterable.next()

show_image(torchvision.utils.make_grid(images))
print(' '.join([f'{classes[labels[j]]:5s}' for j in range(batch_size)]))
```


    
![png](output_10_0.png)
    


    plane plane deer  bird 


looks like utils.make_grid() will take a batch of images and concatenate them along an axis so they're easier to view


```python
torchvision.utils.make_grid(images).shape
```




    torch.Size([3, 36, 138])



# 2. Define a convolutional network


```python
class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # start flattening at dimension 1, which is channel.
        # This preserves the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
```

# 3. Define loss function and optimizer


```python
crit = nn.CrossEntropyLoss()
opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

# 4. Train the network


```python
n_epochs = 2

for epoch in range(2):
    
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader)):
        
        # get inputs
        inputs, labels = data
        
        # zero out gradients
        opt.zero_grad()
        
        # forward pass -> backward pass -> optimize
        outputs = net(inputs)
        loss = crit(outputs, labels)
        loss.backward()
        opt.step()
        
        # calculate statistics
        running_loss += loss.item()
        if i%2000 == 1999: # print every 2000 batches
            print(f"epoch: {epoch+1}, step: {i+1}, avg loss: {running_loss / 2000:.3f}")
            running_loss = 0
```


      0%|          | 0/12500 [00:00<?, ?it/s]


    epoch: 1, step: 2000, avg loss: 1.292
    epoch: 1, step: 4000, avg loss: 1.281
    epoch: 1, step: 6000, avg loss: 1.295
    epoch: 1, step: 8000, avg loss: 1.250
    epoch: 1, step: 10000, avg loss: 1.234
    epoch: 1, step: 12000, avg loss: 1.215



      0%|          | 0/12500 [00:00<?, ?it/s]


    epoch: 2, step: 2000, avg loss: 1.166
    epoch: 2, step: 4000, avg loss: 1.151
    epoch: 2, step: 6000, avg loss: 1.158
    epoch: 2, step: 8000, avg loss: 1.153
    epoch: 2, step: 10000, avg loss: 1.157
    epoch: 2, step: 12000, avg loss: 1.130


Save the trained model


```python
torch.save(net.state_dict(), model_path)
```

# 5. Test on some training data


```python
data_iterable = iter(testloader)
images, labels = data_iterable.next()

# show some images
show_image(torchvision.utils.make_grid(images))
print("GT: ", ' '.join([f'{classes[labels[j]]:.5s}' for j in range(4)]))
```


    
![png](output_22_0.png)
    


    GT:  cat ship ship plane


Load the model


```python
net = Net()
net.load_state_dict(torch.load(model_path))
```




    <All keys matched successfully>




```python
outputs = net(images)

# The outputs of the model are unnormalized probabilities for each class.
# to get the prediction, select the index of the output that has the highest value
_, preds = torch.max(outputs, 1)

print("pred: ", ' '.join([f"{classes[preds[j]]:.5s}" for j in range(4)]))
```

    pred:  cat ship truck plane


Now do a test loop on the whole dataset


```python
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        
        outputs = net(images)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print(f"accuracy on test images = {100* correct // total} %")
```

    accuracy on test images = 59 %


Calculate class-level accuracy


```python
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

```

    Accuracy for class: plane is 70.9 %
    Accuracy for class: car   is 68.1 %
    Accuracy for class: bird  is 48.8 %
    Accuracy for class: cat   is 38.7 %
    Accuracy for class: deer  is 60.4 %
    Accuracy for class: dog   is 48.4 %
    Accuracy for class: frog  is 53.7 %
    Accuracy for class: horse is 67.3 %
    Accuracy for class: ship  is 74.2 %
    Accuracy for class: truck is 62.8 %



```python

```
