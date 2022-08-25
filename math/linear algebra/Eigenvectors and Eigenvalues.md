# Eigenvectors and Eigenvalues


Sources:
https://www.youtube.com/watch?v=PFDu9oVAE-g

similar topics:
- linear systems
- determinates
- change of basis

One way to think about matricies is to picture them as linear transformations. Think about a 2d coordinate system, with $\hat{x}$ and $\hat{y}$ unit vectors. The matrix 
$$
\begin{bmatrix}
3 & 1 \\ 0& 2
\end{bmatrix}
$$
will take the $\hat{x}$ vector (originally $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ ) and map it to the coordinate $\begin{bmatrix} 3 \\ 0 \end{bmatrix}$ in the new coordinate system and take the $\hat{y}$ vector (originally $\begin{bmatrix} 0 \\ 1 \end{bmatrix}$ ) and map it to the coordinate $\begin{bmatrix} 1 \\ 2 \end{bmatrix}$ in the new coordinate system.   

When undergoing this transformation, most vectors in the the space will be rotated and / or stretched / shrinked. 

For example, the vector $\begin{bmatrix} 1 \\ 1 \end{bmatrix}$ will get mapped to $\begin{bmatrix} 4 \\ 2 \end{bmatrix}$:

$$
\begin{bmatrix}
3 & 1 \\ 0& 2
\end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = 

\begin{bmatrix}
3*1 + 1*1 \\ 0*1 + 2*1
\end{bmatrix}

= 

\begin{bmatrix}
4 \\ 2
\end{bmatrix}
$$
However take, for example, the vector$\begin{bmatrix} -1 \\ 1 \end{bmatrix}$:

$$
\begin{bmatrix}
3 & 1 \\ 0 & 2
\end{bmatrix} 

\begin{bmatrix} -1 \\ 1 \end{bmatrix} = 

\begin{bmatrix}
3*-1 + 1*1 \\ 0*1 + 2*1
\end{bmatrix}

= 

\begin{bmatrix}
-2 \\ 2
\end{bmatrix}
$$
**The resultant vector after the linear transformation is a stretched version of the original vector**

The name we give to these special vectors are *eigenvectors* 

The factor by which the eigenvector is streteched is called the *eigenvalue*

### Other discussion points / applications
- Consider a 3d rotation - the axis that does not get rotated during that transformation is (by definition) the axis of rotation. So finding the eigenvector of a 3d rotation gives you the axis of rotation
- The eigenvectors of the covariance matrix of a dataset give the principal components (PCA), where the eigenvalues describe how much variance is in each principal component (i.e. the relative importance of each component)
- usually the eigenvectors / eigenvalues let you get at the heart of what a linear transformation really does

## How to calculate the eigenvectors and eigenvalues

$$
A \vec{v} = \lambda \vec{v}
$$
where $A$ = matrix of interest, $\vec{v}$ = eigenvector, $\lambda$ = eigenvalue

rewrite it as:

$$
A \vec{v} = (\lambda I) \vec{v}
$$
(where $I$ is the identity matrix)

some manipulation:
$$
A \vec{v} - (\lambda I) \vec{v} =  (A - \lambda I) \vec{v} = 0
$$

For this to be true, the linear transformation associated with $A-\lambda I$ must squish $\vec{v}$ to 0. This is equivalent to saying $\text{det}(A-\lambda I)=0$ 