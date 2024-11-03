# Inductance
- inductance is a consequence of Faraday's law of induction:

$$
\int \vec E \cdot dl = - \frac{d \Phi_B}{dt}
$$

Where $\Phi_B$ is the magnetic flux ($\int B \cdot dA$)


## Inductors

Inductors store magnetic fields, in much the same way that capacitors store electric fields.

The unit for an inductor is a Henry:

$$
\begin{aligned}
L &= \text{henry (H)} \\
&= \frac{\text{kg}\cdot\text{m}^2}{\text{s}^2\cdot\text{A}^2} \\
&= V / ( I / t ) \\
&= \Phi / I
\end{aligned}
$$

**Inductance of a solenoid**

since the magnetic field is:
$$
B = \frac{\mu_0 N i}{l}
$$

magnetic flux is:

$$
\Phi = \frac{\mu_0 N i A}{l}
$$

so inductance is:

$$
L = \frac{\mu_0 N^2 A}{l}
$$

**Ohm's law for inductors**

$$
v = L \frac{di}{dt}
$$

where:
- v = instantaneous voltage cross the inductor
- L = inductance in Henries (H)
- $\frac{di}{dt}$ = instantaneous rate of change for current in Amperes per second (A/s)

**Energy inside an inductor**

$$
U = L \int_0^I i di = \frac{1}{2} L I^2
$$


## Charging and discharging RL circuits
![[Pasted image 20240304073827.png]]

For a charging LR circuit (switch connected to batter) - kirchoff's loop rule gives us this:

$$
-iR - L \frac{di}{dt} + \mathcal{E} = 0
$$

Solving for i gives:

$$
i = (\mathcal{E} / R) (1 - e^{t / \tau}) \space \text{where} \space \tau = L/R
$$


For a discharging LR circuit, kirchoff's loop rule gives:

$$
-iR - L\frac{di}{dt} = 0
$$

solving for i gives:

$$
i = (\mathcal{E} / R)(e^{-t/\tau}) \space \text{where} \space \tau = L/R
$$

# References
- http://hyperphysics.phy-astr.gsu.edu/hbase/electric/induct.html
- https://www.usna.edu/Users/physics/jamer/_files/lesson28.pdf
- https://www.allaboutcircuits.com/textbook/direct-current/chpt-15/inductors-and-calculus/
