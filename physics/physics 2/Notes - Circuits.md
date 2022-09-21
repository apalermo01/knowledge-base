# Circuits in a nutshell


# Important quantities

- current - amount of charge flowing through a component or wire (unit = amps = $A$ = coulombs per second = $\frac{C}{s}$)
- resistance - ability of a material to dissipate charge (unit = ohms = $\Omega$ (greek letter omega))
- voltage - potential difference between two points (this is electric potential - $k\frac{q}{r}$) (unit = voltage = $V$)
- capacitance - ability of a material to hold charge (unit = Farad = $F$ = seconds per ohm = $\frac{s}{\Omega}$, base SI untis =$\frac{s^4 A^2}{kg m^2}$)

For more details on units 

# Parallel vs. Series circuits

**Series** circuits have elements in a chain. **Parallel** circuits have branching elements. For more complex circuits, it is not uncommon for some elements to be in series and others to be in parallel

**series circuit**
![[Pasted image 20220920174332.png]]

**Parallel circuit**
![[Pasted image 20220921160007.png]]

When calculating equivalent capacitance or resistance, you will frequently encounter circuits with elements in both series and parallel. In this case, identify 

# Resistance and Capacitance at the microscopic level

**Resistors** 
Think about electricity as water flowing through a pipe. In this analogy, a resistor will be a constriction in the pipe (i.e. it keeps water / electricity from flowing).

![[Pasted image 20220921154416.png]]

In this example, the resistor is a small tube with cross sectional area *A* and length *L*. The resistance of this resistor is modeled by the equation

$$
R = \rho \frac{L}{A}
$$
where $L \text{ and } A$ are the quantities mentioned above and $\rho$ is resistivity,  which is a property of the material that is normally given if needed


**Capacitors**
Think of a parallel plate capacitor. The capacitance, or the ability of the component to store an electric charge, like the resistor, is dictated by the geometry of the component. 

![[Pasted image 20220921155107.png]]

Here we have a parallel plat capacitor where the two plates have an area $A$ and are separated by a distance $d$. The capacitance is defined by:

$$
C = \epsilon \frac{A}{d}
$$
where $\epsilon$ is the permitivity of the material between the two plates. If there is nothing between the two plates (e.g. vacuum or air), then use $\epsilon = \epsilon_0 = 8.85\times10^{-12} \frac{\text{F}}{\text{m}}$. If there is a material between the two plates (called a dielectric), then the permitivity is given by $\epsilon = \epsilon_0 \epsilon_r$, where $\epsilon_r$ is the relative permitivity and is a property of the dielectric.

# Important equations

**Ohm's law**
$$
V = \frac{I}{R}
$$
**Power dissipated**
$$
P = IV = \frac{V^2}{R} = I^2 R
$$
**Voltage, current, resistance, and capacitance in series vs. parallel**
| Series | parallel |
| --------- | -------- |
| $V_{\text{tot}} = V_1 + V_2 + ...$ | $V_{\text{tot}} = V_1 = V_2 = ...$ |
| $I_{\text{tot}} = I_1 = I_2 = ...$ | $I_{\text{tot}}  = I_1 + I_2 + ...$ |
| $R_{\text{tot}} = R_1 + R_2 + ...$ | $\frac{1}{R_{\text{tot}}} = \frac{1}{R_1} + \frac{1}{R_2} + ...$ |
| $\frac{1}{C_{\text{tot}}} = \frac{1}{C_1} + \frac{1}{C_2} + ...$ | $C_{\text{tot}} = C_1 + C_2 + ...$ |



