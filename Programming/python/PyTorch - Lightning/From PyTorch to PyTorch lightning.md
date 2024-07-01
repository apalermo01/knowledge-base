```python
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
```

---
# PyTorch Lightning

1. Model
2. Optimizer
3. Data
4. training loop "the magic"
5. validation loop "the validation magic"


```python
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class ResNet(pl.LightningModule):
  """
  This is exactly the same as an nn module
  
  just with some extra optional ingredients
  
  NOTE: no need for .cuda()- lightning does that for us
  """
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(28*28, 64)
    self.l2 = nn.Linear(64, 64)
    self.l3 = nn.Linear(64, 10)
    self.do = nn.Dropout(0.1)
    self.loss = nn.CrossEntropyLoss()

  def forward(self, x):
    h1 = nn.functional.relu(self.l1(x))
    h2 = nn.functional.relu(self.l2(h1))
    do = self.do(h2+h1)
    logits = self.l3(do)
    return logits
  
  def configure_optimizers(self):
    """
    pl function- can configure as many optimizers as we want
    pl gives us a train loop for each optimizer
    """
    optimizer = optim.SGD(self.parameters(), lr=1e-2)
    return optimizer
  
  ### training loop
  def training_step(self, batch, batch_idx):
    """
    pl function - implements training loop. 
    this is the magic
    """
    x, y = batch
    
    # x: b x 1 x 28 x 28
    b = x.size(0)
    x = x.view(b, -1)
    
    # 1 forward
    logits = self(x) #model(x) # l: logits
    
    # 2 compute objective function
    J = self.loss(logits, y)
    
    # lightning detaches automatically, need to return with graph attached. 
    # return J
  
    # calculate accuracy
    # metrics can be automatically calculated across all gpus for multi-gpu training
    acc = accuracy(logits, y)
    pbar = {'train_acc': acc}

    # equivalently
    # 3 reserved words: 'log', 'loss', 'progress_bar'
    return {'loss': J, 'progress_bar': pbar}

#   def backward(self, trainer, loss, optimizer, optimizer_idx):
#     """
#     This method is implemented for us, but if we want we can override it for custom functionality
#     """
#     loss.backward()

  def train_dataloader(self):
    """
    use this if we need to figure out the number of classes
    """
    train_data = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
    self.train, self.val = random_split(train_data, [55000, 5000])
    train_loader = DataLoader(self.train, batch_size=32)
    #val_loader = DataLoader(val, batch_size=32)
    return train_loader
  
  def val_dataloader(self):

    val_loader = DataLoader(self.val, batch_size=32)
    return val_loader

  ### 2 methods for validation loop: validation_step, 
  def validation_step(self, batch, batch_idx):
    """
    We generally don't want metrics for every batch. plot for whole validation set.
    For every single batch in the validation loop, get the accuracy & loss. Lightning will cache it all for us
    """
    results = self.training_step(batch, batch_idx)
    results['progress_bar']['val_acc'] = results['progress_bar']['train_acc']
    del results['progress_bar']['train_acc']
    return results

  def validation_epoch_end(self, val_step_outputs):
    # [results, results, results, results, ...]
    # calcualte avg val loss for all val outputs
    avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
    avg_acc = torch.tensor([x['progress_bar']['val_acc'] for x in val_step_outputs]).mean()
    # note: early stopping is implemented automatically
    pbar = {'avg_val_acc': avg_val_acc}
    return {'val_loss': avg_val_loss, 'progress_bar': pbar} # val loss is all we care about for early stopping / checkpoint

model = ResNet()
```


```python
trainer = pl.Trainer(progress_bar_refresh_rate=20,
                     max_epochs=5,
                     gpus=1)
trainer.fit(model)
```

    GPU available: True, used: True
    TPU available: False, using: 0 TPU cores
    IPU available: False, using: 0 IPUs
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    
      | Name | Type             | Params
    ------------------------------------------
    0 | l1   | Linear           | 50.2 K
    1 | l2   | Linear           | 4.2 K 
    2 | l3   | Linear           | 650   
    3 | do   | Dropout          | 0     
    4 | loss | CrossEntropyLoss | 0     
    ------------------------------------------
    55.1 K    Trainable params
    0         Non-trainable params
    55.1 K    Total params
    0.220     Total estimated model params size (MB)



    Validation sanity check: 0it [00:00, ?it/s]



    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    /tmp/ipykernel_10853/3806986446.py in <module>
          2                      max_epochs=5,
          3                      gpus=1)
    ----> 4 trainer.fit(model)
    

    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py in fit(self, model, train_dataloaders, val_dataloaders, datamodule, train_dataloader)
        550         self.checkpoint_connector.resume_start()
        551 
    --> 552         self._run(model)
        553 
        554         assert self.state.stopped


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py in _run(self, model)
        920 
        921         # dispatch `start_training` or `start_evaluating` or `start_predicting`
    --> 922         self._dispatch()
        923 
        924         # plugin will finalized fitting (e.g. ddp_spawn will load trained model)


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py in _dispatch(self)
        988             self.accelerator.start_predicting(self)
        989         else:
    --> 990             self.accelerator.start_training(self)
        991 
        992     def run_stage(self):


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/accelerators/accelerator.py in start_training(self, trainer)
         90 
         91     def start_training(self, trainer: "pl.Trainer") -> None:
    ---> 92         self.training_type_plugin.start_training(trainer)
         93 
         94     def start_evaluating(self, trainer: "pl.Trainer") -> None:


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py in start_training(self, trainer)
        159     def start_training(self, trainer: "pl.Trainer") -> None:
        160         # double dispatch to initiate the training loop
    --> 161         self._results = trainer.run_stage()
        162 
        163     def start_evaluating(self, trainer: "pl.Trainer") -> None:


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py in run_stage(self)
        998         if self.predicting:
        999             return self._run_predict()
    -> 1000         return self._run_train()
       1001 
       1002     def _pre_training_routine(self):


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py in _run_train(self)
       1033             self.progress_bar_callback.disable()
       1034 
    -> 1035         self._run_sanity_check(self.lightning_module)
       1036 
       1037         # enable train mode


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/trainer/trainer.py in _run_sanity_check(self, ref_model)
       1116 
       1117             # reload dataloaders
    -> 1118             self._evaluation_loop.reload_evaluation_dataloaders()
       1119 
       1120             # run eval step


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/loops/dataloader/evaluation_loop.py in reload_evaluation_dataloaders(self)
        170             self.trainer.reset_test_dataloader(model)
        171         elif self.trainer.val_dataloaders is None or self.trainer._should_reload_dl_epoch:
    --> 172             self.trainer.reset_val_dataloader(model)
        173 
        174     def on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/trainer/data_loading.py in reset_val_dataloader(self, model)
        439         has_step = is_overridden("validation_step", model)
        440         if has_loader and has_step:
    --> 441             self.num_val_batches, self.val_dataloaders = self._reset_eval_dataloader(model, "val")
        442 
        443     def reset_test_dataloader(self, model) -> None:


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/trainer/data_loading.py in _reset_eval_dataloader(self, model, mode)
        344         # always get the loaders first so we can count how many there are
        345         loader_name = f"{mode}_dataloader"
    --> 346         dataloaders = self.request_dataloader(model, mode)
        347 
        348         if not isinstance(dataloaders, list):


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/pytorch_lightning/trainer/data_loading.py in request_dataloader(self, model, stage)
        482         """
        483         self.call_hook(f"on_{stage}_dataloader")
    --> 484         dataloader = getattr(model, f"{stage}_dataloader")()
        485         if isinstance(dataloader, tuple):
        486             dataloader = list(dataloader)


    /tmp/ipykernel_10853/361698000.py in val_dataloader(self)
         81   def val_dataloader(self):
         82 
    ---> 83     val_loader = DataLoader(self.val, batch_size=32)
         84     return val_loader
         85 


    ~/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/torch/nn/modules/module.py in __getattr__(self, name)
       1128             if name in modules:
       1129                 return modules[name]
    -> 1130         raise AttributeError("'{}' object has no attribute '{}'".format(
       1131             type(self).__name__, name))
       1132 


    AttributeError: 'ResNet' object has no attribute 'val'



```python
! ls lightning_logs/
```

    version_0  version_1  version_2  version_3  version_4  version_5


lightning saved the best checkpoint for us plus logs


```python

```


```python
class ImageClassifier(nn.Module):
  def __init__(self):
    self.resnet = ResNet()
```


```python
# train, val split
train_data = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)
```

    /home/alex/anaconda3/envs/pl_lightning/lib/python3.9/site-packages/torchvision/datasets/mnist.py:498: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /opt/conda/conda-bld/pytorch_1631630841592/work/torch/csrc/utils/tensor_numpy.cpp:180.)
      return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)



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

    Epoch 1, training loss: 0.84, training accuracy: 0.78
    Epoch 1, validation loss: 0.40, validation accuracy: 0.89
    Epoch 2, training loss: 0.38, training accuracy: 0.89
    Epoch 2, validation loss: 0.33, validation accuracy: 0.91
    Epoch 3, training loss: 0.31, training accuracy: 0.91
    Epoch 3, validation loss: 0.28, validation accuracy: 0.92
    Epoch 4, training loss: 0.27, training accuracy: 0.92
    Epoch 4, validation loss: 0.25, validation accuracy: 0.93
    Epoch 5, training loss: 0.24, training accuracy: 0.93
    Epoch 5, validation loss: 0.23, validation accuracy: 0.93



```python

```
