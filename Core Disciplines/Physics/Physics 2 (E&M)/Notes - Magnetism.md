# Key ideas for magnetism


## Units
SI Unit for magnetic field strength is the Tesla ($T$) 
$$
T = \frac{kg}{s^2A}
$$
can also use Gauss (G): $1G = 10^{-4} T$

## Magnetic forces

**Force on a moving charge**

$$
\vec F = \vec E + q \vec v \times \vec B
$$
$$
|F| = q|v||B|\sin(\theta)
$$
where $\times$ is the cross product (see below)

**Force on a current carrying wire**
$$
\vec F = I \vec L \times \vec B
$$
$$
|F| = I|L||B|\sin(\theta)
$$


**Force between 2 parallel current carrying wires**

$$
F = I_1 l B_2 = I_1 l \frac{\mu_0 I_2}{2 \pi d} = \frac{\mu_0 l I_1 I_2}{2 \pi d}
$$

## Working with cross products

In progress


## Biot-savart law

- The Biot-savart law is like coulomb's law for magnetism. Coulomb's law gives the electric field due to a point charge. The Biot-savart law gives the magnetic field due to a "point current". 

$$
d\vec B = \frac{\mu_0}{4 \pi} \frac{I d\vec s \times \hat r}{r^2}
$$

- $I$ is the current through a segment of wire
- $d \vec s$  is an incremental step along the wire
- $r$ and $\hat r$ is the distance and direction, respectively, from the "point current" to the point of interest

To solve problems related to the biot-savart law, integrate along the path that the current is taking.


## Ampere's law

- Ampere's law is just Gauss's law for magnetism. Instead of directly integrating over all "point currents" in the area of interest, you integrate over some space around all the currents using a line integral.

$$
\oint \vec B \cdot d \vec s = \mu_0 I_{\text{enclosed}}
$$

## References
- Physics for scientists and engineers, 4th edition, Knight
