# Profiling pytorch models

https://pytorch.org/tutorials/beginner/profiler.html

Identifies time and memory costs of pytorch operations. can be printed as a table or returned in JSON


```python
import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler
```

define a simple model


```python
class MyModule(nn.Module):
  def __init__(self,
               in_features: int,
               out_features: int,
               bias: bool = True):
    super(MyModule, self).__init__()
    self.linear = nn.Linear(in_features, out_features, bias)
    
  def forward(self, inputs, mask):
    with profiler.record_function("LINEAR PASS"):
      out = self.linear(inputs)
    
    with profiler.record_function("MASK INDICES"):
      threshold = out.sum(axis=1).mean().item()
      hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
      hi_idx = torch.from_numpy(hi_idx).cuda()

    return out, hi_idx
```

## profile forward pass

initialize input, mask, and model

warm-up cuda (for accuracy), then do forward pass in context manager

`with_stack=True` - appends file and line number of operation in trace


```python
model = MyModule(100, 10).cuda()
inputs = torch.rand(32, 100).cuda()
mask = torch.rand((100, 100, 100), dtype=torch.double).cuda()

# warm up 
model(inputs, mask)

with profiler.profile(with_stack=True, profile_memory=True) as prof:
  out, idx = model(inputs, mask)
```

## Printing profiler results

`profiler.key_averages`- agg results by name, input shapes, and/or stack traces. 

Use grou_by_stack_n = 5 to aggregate by operation and traceback


```python
print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=5))
```

    -----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
                             Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  Source Location                                                              
    -----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
                      aten::zeros        90.70%     224.923ms        90.80%     225.173ms     225.173ms           4 b           0 b           0 b           0 b             1  ..._3/lib/python3.7/site-packages/torch/autograd/profiler.py(611): __init__  
                                                                                                                                                                               /tmp/ipykernel_42446/2206843926.py(10): forward                              
                                                                                                                                                                               ...lib/python3.7/site-packages/torch/nn/modules/module.py(1051): _call_impl  
                                                                                                                                                                               /tmp/ipykernel_42446/4099370643.py(9): <module>                              
                                                                                                                                                                               ...python3.7/site-packages/IPython/core/interactiveshell.py(3441): run_code  
                                                                                                                                                                                                                                                            
                     MASK INDICES         5.25%      13.010ms         8.79%      21.809ms      21.809ms          -4 b      -7.63 Mb      14.22 Mb      -1.00 Kb             1  ...3/lib/python3.7/site-packages/torch/autograd/profiler.py(614): __enter__  
                                                                                                                                                                               /tmp/ipykernel_42446/2206843926.py(13): forward                              
                                                                                                                                                                               ...lib/python3.7/site-packages/torch/nn/modules/module.py(1051): _call_impl  
                                                                                                                                                                               /tmp/ipykernel_42446/4099370643.py(9): <module>                              
                                                                                                                                                                               ...python3.7/site-packages/IPython/core/interactiveshell.py(3441): run_code  
                                                                                                                                                                                                                                                            
                      aten::copy_         2.14%       5.317ms         2.14%       5.317ms       5.317ms           0 b           0 b           0 b           0 b             1  /tmp/ipykernel_42446/2206843926.py(16): forward                              
                                                                                                                                                                               ...lib/python3.7/site-packages/torch/nn/modules/module.py(1051): _call_impl  
                                                                                                                                                                               /tmp/ipykernel_42446/4099370643.py(9): <module>                              
                                                                                                                                                                               ...python3.7/site-packages/IPython/core/interactiveshell.py(3441): run_code  
                                                                                                                                                                               ...n3.7/site-packages/IPython/core/interactiveshell.py(3361): run_ast_nodes  
                                                                                                                                                                                                                                                            
                      aten::copy_         1.05%       2.615ms         1.05%       2.615ms       2.615ms           0 b           0 b           0 b           0 b             1  /tmp/ipykernel_42446/2206843926.py(15): forward                              
                                                                                                                                                                               ...lib/python3.7/site-packages/torch/nn/modules/module.py(1051): _call_impl  
                                                                                                                                                                               /tmp/ipykernel_42446/4099370643.py(9): <module>                              
                                                                                                                                                                               ...python3.7/site-packages/IPython/core/interactiveshell.py(3441): run_code  
                                                                                                                                                                               ...n3.7/site-packages/IPython/core/interactiveshell.py(3361): run_ast_nodes  
                                                                                                                                                                                                                                                            
                      aten::addmm         0.08%     189.347us         0.13%     314.357us     314.357us           0 b           0 b       1.50 Kb       1.50 Kb             1  ...da_IQ_3/lib/python3.7/site-packages/torch/nn/functional.py(1847): linear  
                                                                                                                                                                               ...IQ_3/lib/python3.7/site-packages/torch/nn/modules/linear.py(96): forward  
                                                                                                                                                                               ...lib/python3.7/site-packages/torch/nn/modules/module.py(1051): _call_impl  
                                                                                                                                                                               /tmp/ipykernel_42446/2206843926.py(11): forward                              
                                                                                                                                                                               ...lib/python3.7/site-packages/torch/nn/modules/module.py(1051): _call_impl  
                                                                                                                                                                                                                                                            
    -----------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ---------------------------------------------------------------------------  
    Self CPU time total: 247.988ms
    



```python

```
