# Introduction to TorchScript

https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html

language-independent representation of pytorch models that can be run in other environments (e.g. C++)


```python
import torch
print(torch.__version__)
```

    1.7.1


## Basics of PyTorch Model Authoring

define a simple module


```python
class MyCell1(torch.nn.Module):
  def __init__(self):
    super(MyCell1, self).__init__()
    
  def forward(self, x, h):
    new_h = torch.tanh(x+h)
    return new_h, new_h
```


```python
my_cell = MyCell1()

x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))
```

    (tensor([[0.4333, 0.4071, 0.2366, 0.7620],
            [0.7723, 0.6108, 0.8062, 0.7892],
            [0.7976, 0.7747, 0.5746, 0.8612]]), tensor([[0.4333, 0.4071, 0.2366, 0.7620],
            [0.7723, 0.6108, 0.8062, 0.7892],
            [0.7976, 0.7747, 0.5746, 0.8612]]))


Something a little more interesting


```python
class MyCell2(torch.nn.Module):
  def __init__(self):
    super(MyCell2, self).__init__()
    self.linear = torch.nn.Linear(4, 4)
    
  def forward(self, x, h):
    new_h = torch.tanh(self.linear(x)+h)
    return new_h, new_h
```


```python
my_cell = MyCell2()

x = torch.rand(3, 4)
h = torch.rand(3, 4)
print(my_cell(x, h))
```

    (tensor([[ 0.3396,  0.6812, -0.0612,  0.0077],
            [ 0.7622,  0.2133, -0.2595,  0.6893],
            [ 0.8999,  0.6787,  0.4849,  0.5211]], grad_fn=<TanhBackward>), tensor([[ 0.3396,  0.6812, -0.0612,  0.0077],
            [ 0.7622,  0.2133, -0.2595,  0.6893],
            [ 0.8999,  0.6787,  0.4849,  0.5211]], grad_fn=<TanhBackward>))


More complicated cell


```python
class MyDecisionGate(torch.nn.Module):
  def forward(self, x):
    if x.sum() > 0:
      return x
    else:
      return -x
    
class MyCell3(torch.nn.Module):
  def __init__(self):
    super(MyCell3, self).__init__()
    self.dg = MyDecisionGate()
    self.linear = torch.nn.Linear(4, 4)
    
  def forward(self, x, h):
    new_h = torch.tanh(self.dg(self.linear(x))+h)
    return new_h, new_h
```


```python
my_cell = MyCell3()
print(my_cell)
print(my_cell(x, h))
```

    MyCell3(
      (dg): MyDecisionGate()
      (linear): Linear(in_features=4, out_features=4, bias=True)
    )
    (tensor([[-0.0047,  0.4234, -0.0544,  0.3878],
            [ 0.5767, -0.0541, -0.0144,  0.8046],
            [ 0.7059,  0.6536,  0.5000,  0.7352]], grad_fn=<TanhBackward>), tensor([[-0.0047,  0.4234, -0.0544,  0.3878],
            [ 0.5767, -0.0541, -0.0144,  0.8046],
            [ 0.7059,  0.6536,  0.5000,  0.7352]], grad_fn=<TanhBackward>))


### Tracing Modules


```python
my_cell = MyCell2()

x, h = torch.rand(3, 4), torch.rand(3, 4)
traced_cell = torch.jit.trace(my_cell, (x, h))
print(traced_cell)
traced_cell(x, h)
```

    MyCell2(
      original_name=MyCell2
      (linear): Linear(original_name=Linear)
    )





    (tensor([[ 0.0605,  0.4208,  0.2279, -0.5719],
             [-0.5544, -0.1919,  0.0962, -0.0751],
             [ 0.5301,  0.5226,  0.7805,  0.5141]], grad_fn=<TanhBackward>),
     tensor([[ 0.0605,  0.4208,  0.2279, -0.5719],
             [-0.5544, -0.1919,  0.0962, -0.0751],
             [ 0.5301,  0.5226,  0.7805,  0.5141]], grad_fn=<TanhBackward>))



This "traced" a sample input through the model and recorded what operations took place


```python
print(traced_cell.graph)
```

    graph(%self.1 : __torch__.MyCell2,
          %input : Float(3:4, 4:1, requires_grad=0, device=cpu),
          %h : Float(3:4, 4:1, requires_grad=0, device=cpu)):
      %19 : __torch__.torch.nn.modules.linear.Linear = prim::GetAttr[name="linear"](%self.1)
      %21 : Tensor = prim::CallMethod[name="forward"](%19, %input)
      %12 : int = prim::Constant[value=1]() # <ipython-input-12-7b94133aa41b>:7:0
      %13 : Float(3:4, 4:1, requires_grad=1, device=cpu) = aten::add(%21, %h, %12) # <ipython-input-12-7b94133aa41b>:7:0
      %14 : Float(3:4, 4:1, requires_grad=1, device=cpu) = aten::tanh(%13) # <ipython-input-12-7b94133aa41b>:7:0
      %15 : (Float(3:4, 4:1, requires_grad=1, device=cpu), Float(3:4, 4:1, requires_grad=1, device=cpu)) = prim::TupleConstruct(%14, %14)
      return (%15)
    



```python
print(traced_cell.code)
```

    def forward(self,
        input: Tensor,
        h: Tensor) -> Tuple[Tensor, Tensor]:
      _0 = torch.add((self.linear).forward(input, ), h, alpha=1)
      _1 = torch.tanh(_0)
      return (_1, _1)
    


calling traced_cell produces the same result as the original module

## Using scripting to convert modules


```python
class MyDecisionGate(torch.nn.Module):
  def forward(self, x):
    if x.sum() > 0:
      return x
    else:
      return -x
    
class MyCell3(torch.nn.Module):
  def __init__(self, dg):
    super(MyCell3, self).__init__()
    self.dg = dg
    self.linear = torch.nn.Linear(4, 4)
    
  def forward(self, x, h):
    new_h = torch.tanh(self.dg(self.linear(x))+h)
    return new_h, new_h
  
my_cell = MyCell3(MyDecisionGate())
traced_cell = torch.jit.trace(my_cell, (x, h))

print(traced_cell.dg.code)
print(traced_cell.code)
```

    def forward(self,
        argument_1: Tensor) -> Tensor:
      return torch.neg(argument_1)
    
    def forward(self,
        input: Tensor,
        h: Tensor) -> Tuple[Tensor, Tensor]:
      _0 = (self.dg).forward((self.linear).forward(input, ), )
      _1 = torch.tanh(torch.add(_0, h, alpha=1))
      return (_1, _1)
    


    <ipython-input-20-72d46ebe4639>:3: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if x.sum() > 0:


We lost one of the decision branches becuase it was never executed!

To fix this, use a script compiler


```python
scripted_gate = torch.jit.script(MyDecisionGate())

my_cell = MyCell3(scripted_gate)
scripted_cell = torch.jit.script(my_cell)

print(scripted_gate.code)
print(scripted_cell.code)
```

    def forward(self,
        x: Tensor) -> Tensor:
      _0 = bool(torch.gt(torch.sum(x, dtype=None), 0))
      if _0:
        _1 = x
      else:
        _1 = torch.neg(x)
      return _1
    
    def forward(self,
        x: Tensor,
        h: Tensor) -> Tuple[Tensor, Tensor]:
      _0 = (self.dg).forward((self.linear).forward(x, ), )
      new_h = torch.tanh(torch.add(_0, h, alpha=1))
      return (new_h, new_h)
    


Can combine scripting and tracing (do this later)
