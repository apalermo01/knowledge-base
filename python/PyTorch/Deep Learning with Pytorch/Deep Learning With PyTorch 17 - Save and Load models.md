# Deep Learning With PyTorch 17 Save and Load models


```python
import torch
import torch.nn as nn

### Complete model
torch.save(model, PATH)

# model class must be defined somewhere
model = torch.load(PATH)
mode.eval
```

Second option, recommended way


```python
torch.save(model.state_dict(), PATH)

# model must be created again with parameters
model = Model(*args, **kwargs)
model.load_state_dict(torch.load(PATH))
model.eval()
```
