---
tags:
  - neural_networks
  - deep_learning
  - machine_learning
---
LSTM stands for Long Short-Term Memory. They are designed to make inferences off of sequences while holding a better "memory" of previous observations than [[RNN Layers |Recurrent Neural Networks]]. 

I like to think of LSTMs as having 2 "tracks". The first track ($h_{t-1}$) stores information about the previous result. The second track ($C_{t-1}$) stores the "memory" about previous observations. as the LSTM learns, it learns which information to add or forget on this $C$ track.


I've broken down the LSTM algorithm into 11 steps according to the figure below:

![[LSTM diagram.png]]

1. Take 2 inputs: the current value in the sequence and the previous value of the LSTM
2. Steps 2-3 represent the forget gate. concatenate these two inputs and run them through a sigmoid function. The output of this value represents how much to "forget" in the memory ($C$) track.
3. Step 2 outputs a scalar that quantifies how much should be forgotten in the $C$ track. Multiply this scalar by the value of the memory ($C$) track.
4. Steps 4-7 represent the remember gate. First, run the concatenated previous output and current value through another sigmoid to quantify how much of this step to remember (this outputs a scalar)
5. run the concatenated previous output and current value through hyperbolic tangent. This is just an activation function. This can be tweaked as a hyperparameter but $\tanh$ seems to be the convention.
6. Multiply the output of 5 by the scalar we got in 4 - this is how much of the input should be remembered by the model.
7. Add what we got in 6 to the memory ($C$) track
8. The next step is to compute the actual output for this iteration. First, run the current value through a sigmoid to figure out what to output
9. Then, run the memory ($C$) track through an activation function
10. Multiply that output with how much we want to keep of the current input to generate the output
11. The model has 2 outputs that go in 3 different places. $h_t$ serves as both the output for the current node and one of the inputs for the next node. $C_t$ is the output of the model's "memory" for the next node.

## References
- https://colah.github.io/posts/2015-08-Understanding-LSTMs/