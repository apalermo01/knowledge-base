# Fundamentals of Neural Networks

resources / articles:<br> 
https://towardsdatascience.com/what-is-a-perceptron-210a50190c3b<br>
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6<br>
https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi<br>
https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html

## Perceptron

The perceptron is the "neuron" of the neural network. In it's most basic form, it takes on (or more) inputs and multiplies by a set of weights (i.e. it's linear regression)

$$
y = \omega_1 x_1 + \omega_2 x_2 + ...
$$

Perceptrons can also have a bias associated with them:

$$
y = \omega_1 x_1 + \omega_2 x_2 + ... + b
$$

![image.png](attachment:image.png)

## Activation Functions

To make a deep neural network, we need to stack layers of perceptrons. This creates a problem: pushing data through sequential perceptrons is mathematically equivalent to another linear function. In order to introduce nonlinearity, we need an activation function. 

So the output of the perceptron + activation function looks like this:

$$
y_a = act(\omega_1 x_1 + \omega_2 x_2 + ... + b)
$$

where $act()$ is some activation function

some common activation functions:

![image.png](attachment:image.png)

## Objective function

When building and training a neural network, we need a way of "teaching" the network what we want it to learn. Take, for example, handwritten digits. The MNIST dataset consists of 28x28 grayscale images of handwritten digits. To make a classifier, we can build a 3 layer network with the following layers:

- layer 1: 784 perceptrons
- ReLU activation
- layer 2: 512 perceptrons
- ReLU activation
- layer 3: 10 perceptrons

layer 1 has 1 perceptron for each pixel in the input image. Layer 2 is called the hidden layer - it can have any number of perceptrons. If this is a fully connected layer, then every perceptron in layer 2 takes an input from every perceptron in layer 1. This means that each perceptron in layer 2 generates an output based on a linear combination of 784 inputs. Finally, layer 3 contains 10 outputs, each neuron for each possible output. 

For a perfect model, the number i will cause the ith output neuron to have a high output, and all the others 0. 

For example, if the input was the number 1, a perfect model will give this as an output:<br> 
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

If the input was the number 5, this would be the perfect output:<br>
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]

In order to train the model to give us this output, we run the output and the ground truth (what number we actually have) through an objective function. One such objective function for this task is the cross-entropy loss:

$$
-\sum_{c=1}^{M} y_{o, c} log(p_{o, c})
$$

Where: 
- M = number of classes
- $log$ = natural log
- y = 1 if c is the correct classification for observation 0, 0 otherwise
- p = predicted probability that observation o is of class c

(source: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html)

In simple terms, **this is a function that tells the neural network how wrong its prediction is**

## Backpropagation


Now that we have an idea of how wrong the model is, we need to update the model parameters so that it won't be as wrong when it sees the next example. \[incomplete]
