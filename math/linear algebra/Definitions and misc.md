## Span

The span of a set $S$ of vectors, denoted $\text{span}(S)$ is the set of all linear combinations of those vectors

Take 2 vectors [0, 1] and [2, 3]. The span of these 2 vectors is the set of all vectors that satisfy the equation:

$$
\begin{bmatrix}
a \\ b
\end{bmatrix} = 
c_1 \begin{bmatrix}
0 \\ 1
\end{bmatrix} +
c_2 \begin{bmatrix}
2 \\ 3
\end{bmatrix}
$$

where $c_1$ and $c_2$ are real numbers.

If you're asked whether or not a vector is in the span of a set of vectors, set up a system on linear equations and try to solve for $c_1, c_2, ...$. If it is not possible, then that vector is not in the span.

# References
- https://math.oit.edu/~watermang/math_341/341_ch8/F13_341_book_sec_8-1.pdf (also contains some exercises using span)