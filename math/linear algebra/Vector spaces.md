
Put simply, a vector space is a collection of any kind of mathematical objects that can be added and multiplied together. 

## Motivation

All of these problems can be solved using the same toolkit:

$$
\begin{bmatrix}
3 & 2 & 0 \\
1 & 0 & 1 \\
2 & 3 & 8
\end{bmatrix}

\begin{bmatrix}
x_1 \\ x_2 \\ x_3
\end{bmatrix}

= 

\begin{bmatrix}
8 \\ 2 \\ 7
\end{bmatrix}
$$

$$
x_1(3t^2 + 5t - 2) + x_2(0t^2 - t + 6) + x_3 (9t^2 + 0t + 1) = 6t^2 + 9t + 2
$$

$$
x_1 \sin(\pi t) + x_2 \sin(2\pi t) + x_3 \sin(3 \pi t) + ... = e^{5it}
$$

In the first case, we're dealing with column matrices, in the second, polynomials, and the third, functions. 

**Definition: Field**

A **Field** is a set $\mathbb{F}$ of numbers with the property that if $a, b \in \mathbb{F}$, then $a+b$, $a-b$, $ab$, and $a/b$ are also in $\mathbb{F}$. 

With that being said, a **Vector Space** consists of a set $V$, field $\mathbb{F}$, and two operations: 
	- *addition*, which takes two vectors $v, w \in V$ and produces a third vector $v+w \in V$. 
	- *scalar multiplication*, which takes a scalar $c \in \mathbb{F}$ and a vector $v \in V$ and produces a new vector, $cv \in V$. 

and also satisfies the following axioms*:
- addition is associative
- zero vector ($u + 0 = u$)
- existance of negatives ($u + (-u) = 0$)
- multiplication is associative
- multiplication is distributive
- unitary: $1u = u$

\*Note: I am giving a very abbreviated description of the axioms compared to what you would find in a textbook

## Null space

Consider a matrix $A$:

$$
\begin{bmatrix}
x_{11} & ... & x_{1n} \\
... & ... & ... \\
x_{m1} & ... & ... x_{mn}
\end{bmatrix}
$$

The null space of $A$ is the set of all vectors $B$ such that $AB = 0$

In other words, solve the equation for $B$:

$$
\begin{bmatrix}
x_{11} & ... & x_{1n} \\
... & ... & ... \\
x_{m1} & ... & ... x_{mn}
\end{bmatrix}

\begin{bmatrix}
b_1 \\ . \\ . \\  b_n
\end{bmatrix}

= 0
$$

# References

- https://www.math.toronto.edu/gscott/WhatVS.pdf
- https://brilliant.org/wiki/vector-space/