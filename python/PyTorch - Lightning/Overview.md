# PyTorch lightning training intro

https://www.pytorchlightning.ai/tutorials

- LightningModule - full DL training system (e.g. model, collection of models (GAN), etc.)
    - groups everything together into one self-contained system
    - collection of methods
- see examples in repo for use-cases
- autoencoder: encoder and decoder (2 networks)

**defining the module:**<br> 
- initialize model(s) in init
- in forward method, define how it will work in inference
- train_step: define what happens in train step
- validation_step, test_step: do same thing
- configure_optimizers: defines optimizer

No looping over epochs or datasets, trainer does that for us

**Defining dataset**<br> 
- works directly with dataloaders
- can always modify a hook / callback to modify functionality to make it do what you need. 

need 3 things: LightningModule, DataSet, Trainer. 

# Automatic Batch Size Finder

- `auto_scale_batch_size` - finds largest batch size that can fit on GPU
- pass this flag when initializing trainer
- must have `self.batch_size` or `self.save_hyperparams()` in module
- can hard code this into DataLoader after first pass

# Automatic Learning Rate Finder

- most important hyperparameter
- `auto_lr_find` - outputs lr vs loss
    - note: we want the lr where this graph is the steepest
- configure_optimizer won't be called until needed for training (lazy initialization)
- run `trainer.tune(model)` to find best learning rate
- then run training with that initial lr
- only works with single optimizer for now
- "learning rate may be impossible to find with some problems" - can use as ballpark estimate

# Exploding and Vanishing Gradients

- `track_grad_norm` - set to l(n) norm to track norm of the lr throughout training to look for exploding / vanishing gradients. 
- `gradient_clip_val` - keeps exploding gradients from exploding to infinity

# Truncated Back-Propogation Through Time

- `truncated_bptt_steps` - for long sequences, we can backprogate through a small subset of the of the network at a time
- set flag to number of steps

# Reload DataLoaders Every Epoch

- `reload_dataloaders_every_epoch` - loads dataset once by default. Setting this to True re-loads the data every time (use if data is changing or model is deployed in production).

# Lightning Callbacks

- arbitrary self-contained programs.
- gives full control over any section of training. 
- ex: early stopping, saving model checkpoints
- can pass in callbacks when initializing trainer
- specify which hook in which the callback gets called.
- **hook** - a function that will be called at some point during training (over 40). 
- common callbacks found in PytorchLightning- bolts

# Lightning Early Stopping

- automatically stops training once some metric has stopped improving
- default: happens when there has been no improvement in 3 evaluations. 

# Lightning Weights Summary

- `weights_summary` - prints a summary of weights
- set to 'full' for full description of weights (similar to torchinfo)

# Lightning Progress Bar

- `progress_bar_refresh_rate` - how often bar refreshes

# Lightning Profiler

- identifies bottlenecks in training. gives high level description of method calls and how long those calls took. 

# Controlling Lightning Training and Eval Loops

- `min_epochs`, `max_epochs` flags
- `min_steps`, `max_steps` (mini-batch steps)- validation steps do not count
- `check_val_every_n_epochs` - how many train epochs to run before doing the validation step
- `check_val_interval` - runs through validation loop within an epoch (good for cases where there is a very long training loop). 
- `num_sanity_val_steps` - will run this many rounds of validation before training starts (good for debugging). Runs 2 automatically, set to -1 to run entire val loop
- `limit_train_batches`, `limit_val_batches`, `limit_train_batches` - shorten length of loops so we can debug val or test loop without having to wait for train loop to finish

# Training on GPUs

- core: massive scaling w/ GPUs
- PL can train on GPU or CPU with no changes in code
- todo to get this hardware agnosticism: 
    - delete .cuda(), .to('cuda'), etc.
    - when initializing tensors, always use `device = self.device`
    - can also use type_as
- use `gpus` flag to specify number of gpus and/or which specific GPUs to use
    - -1 for all available GPUs
- `log_gpu_memory` - logs memory usage per gpu (uses nvidia-smi under the hood). Turn it on when testing
- `benchmark` - can result in speed-ups in input size is the same
- `deterministic` - removes randomness from training (False by default)

# Training on multiple GPU nodes

- single machine = node
- `num_nodes` - specify number of nodes (computers) to use
    - effective batch size = batch_size * num nodes * num gpus
- many algos that can sync gradients
- default: DistributedDataParallel (implemented by pytorch)
    - each GPU gets a fraction of the dataset and a copy of the model
    - perform forward pass in each GPU
    - then sync the gradients between nodes
    - then each optimizer in each node performs gradient descent
    - IT IS VERY IMPORTANT TO INITIALIZE WITH A SEED (since each copy of the model is updated individually)
    - use DDP whenever possible
    - but it's not supported in windows
    - can use DataParallel if no other option
    - lightning also has ddp2 (see docs)
- `ddp_cpu` - simulates ddp in a cpu for debugging. set num_processes to num gpus to simulate.

# Training on TPUs

- Tensor Processing Units. Optimized for 128x128 matrix multiplication
- 1 TPU = 4 V100s
- use `tpu_cores` flag (no code changes needed)
- still experimental

# Debugging Flags

- `fast_dev_run` - touches every piece of code in model, train, val, and test loop for debugging. No logs, no checkpoints. 
- `overfit_batches` - take a single batch of train data and overfit it as much as possible. 

# Accumulating Gradients

- `accumulate_grad_batches` - simulate a large batch by running several forward steps and accumulate the gradient so that it sees enough examples per batch. 
- can customize how much to accumulate

# Mixed Precision Training

- many models can train with a combo of 32 and 16 bit precision without loosing accuracy
- reduces memory requirements
- can get a speedup on some GPUs
- PyTorch has automatic mixed precision support
- `precision` - switch between 16 and 32 bit precision
- `amp_level` - if using apex library, can set different modes for mixed precision.
