
## Cone

In cartesian coordinates, a cone opening along the z axis with a height $h$ and radius $a$ (specifically, the radius of the open end) is given by:
$$
a^2 z^2 = h^2 x^2 + h^2 y^2
$$
![[Pasted image 20221114110305.png]]
**cartesian**
Solve the equation for z:
$$
\begin{align}
a^2 z^2 &= h^2 x^2 + h^2 y^2 \\
z^2 &= \frac{h^2}{a^2} (x^2 + y^2) \\
z &= \frac{h}{a} \sqrt{x^2 + y^2}
\end{align}
$$
This line (surface) represents the bottom bound of $z$ given an x and y coordinate, therefore the boundary in the z-direction is:

$$
\frac{h}{a} \sqrt{x^2 + y^2} \le z \le h
$$

This z-integral will be the inner integral. After we integrate over z, we can stop thinking about any z-dependancy. 

Now, to get the bounds for x and y, project the maximum bounds for x and y onto the x-y plane -> effectively, this means projecting the circle representing the 'opening' (circle of radius a) of the cone onto the x-y plane
![[Pasted image 20221114110519.png]]\

![[Pasted image 20221114110542.png]]

The equation of this circle is $x^2 + y^2 = a^2$. Start with integrating along y - solve this equation for x so we have $y = \sqrt{a^2 - x^2}$ , so our y-bound will go from the bottom of the circle to the top, so y will go from $-\sqrt{a^2 - x^2}$ to $+\sqrt{a^2 - x^2}$ , so the bound for y is 
$$
-\sqrt{a^2 - x^2} \le y \le +\sqrt{a^2 - x^2} 
$$

leaving the bounds for x to be: 
$$
-a \le x \le a
$$

Now we solve the integral:

![[Pasted image 20221114112038.png]]

**cylindrical**

**spherical**


## Paraboloid

**cartesian**

**cylindrical**

**spherical**



## References
- https://sites.math.washington.edu/~aloveles/Math324Fall2013/f13m324TripleIntegralExamples.pdf