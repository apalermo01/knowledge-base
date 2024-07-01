
## Constants

- $k = 9 \times 10^9 \text{N} \frac{\text{m}^2}{\text{C}^2}$ 
- $\epsilon_0 = 8.85 \times 10^{-12} \frac{\text{C}^2}{\text{Nm}^2}$ (permittivity of free space)

## List of units

- $\text{C}$ = Coulomb = charge
- $\text{N}$ = Newton = force
- $\text{J}$ = Joule = energy
- $\text{V}$ = Volt = electric potential
	- $\text{V} = \frac{\text{kg} \cdot \text m^2}{\text s^3 \cdot \text A} = \frac{\text J}{\text C} = \text A \cdot \Omega = \frac{\text{Wb}}{\text s} = \frac{\text W}{\text A}$ 
- $\text{A}$ = Ampere = current
	- $\text A = \frac{\text C}{\text s}$
- $\Omega$  = Ohm = resistance
	- $\Omega = \frac{\text{kg} \cdot \text m ^2}{\text s \text C^2}\frac{\text V}{\text A} = \frac{\text{W}}{\text A ^2} = \frac{\text V ^2}{\text W} = \frac{\text s}{\text F} = \frac{\text H}{\text s} = \frac{\text J \text s}{\text C ^2} = \frac{\text J}{\text s \cdot \text A ^2} = \frac{\text{kg} \cdot \text m^2}{\text s ^2 \cdot \text A ^2}$
- $\text{F}$ = Farad = capacitance
	- $\text F = \frac{\text s ^2 \cdot \text C ^2}{\text m ^2 \cdot \text{kg}} = \frac{\text C}{\text V} = \frac{\text s ^4 \cdot \text A ^2}{\text m ^2 \cdot \text{kg}} = \frac{\text A \cdot \text s}{\text V} = \frac{\text W \cdot \text s}{\text V^2} = \frac{\text J}{\text V^2} = \frac{\text N \cdot \text m}{\text V ^2} = \frac{\text C ^2}{\text J} = \frac{\text C^2}{\text J} = \frac{\text s}{\Omega} = \frac{\text s ^2}{\text H}$
- $\text{T}$ = Tesla = magnetic field strength
	- $\text{T} = \frac{\text N \cdot \text s }{\text{C} \cdot \text{m}} = \frac{\text{Wb}}{\text{m}^2} = \frac{\text{kg}}{\text A \cdot \text s ^2} = \frac{\text{N}}{\text A \cdot \text m} = \frac{\text J}{\text A \cdot \text m ^2} = \frac{\text V \cdot \text s}{\text m ^2}$ 
- $\text{G}$ = Gauss = another measure for magnetic field strength
	- $1 \text G = 10^{-4} \text{T}$
- $\text{Wb}$ = Weber = magnetic flux density
	- $\text{Wb} = \text T \cdot \text m ^2$
- $H$ = Henry = Inductance
	- $\text H = \frac{\text{kg} \cdot \text m^2}{\text s ^2 \cdot \text A ^2} = \frac{\text V \cdot \text s}{\text A} = \frac{\text N \cdot \text m}{\text A ^2} = \frac{\text s ^2}{\text F} = \Omega \cdot \text s$
## Detailed list of quantities

**Force**
Definition in the context of electric charges:
$$
F = \frac{1}{4 \pi \epsilon_0} \frac{q_1 q_2}{r^2} \hat r = k \frac{q_1 q_2}{r^2}
$$

relationship to other quantities:
$$
\begin{align}
F &= Eq \\
& \\
F &= \frac{\text{E.P.E}}{d} = \frac{U}{d}

\end{align}
$$

Units
$$
\text{N} = \text{Newton} = \text{kg} \frac{\text{m}}{\text{s}^2}
$$
**Electric Field**
Definition

$$
E = \frac{1}{4 \pi \epsilon_0} \frac{q}{r^2} \hat r = k \frac{q}{r^2} \hat r
$$
relationship to other quantities:
$$
\begin{align}
E &= \frac{F}{q} \\
\\
E &= \frac{V}{d}
\end{align}
$$

Units

$$
\frac{\text{N}}{\text{C}} = \frac{\text{V}}{\text{m}}
$$


**Energy (aka Electric potential energy)**

Definition in the context of electric charge
$$
U = \frac{1}{4 \pi \epsilon_0} \frac{q_1 q_2}{r} = k \frac{q_1 q_2}{r}
$$

Relationship to other quantities

$$
\begin{align}
U &= F \cdot d \\
\\
U &= V \cdot q
\end{align}
$$

Units

$$
\text {J} = \text{kg} \frac{\text{m}^2}{\text{s}^2}
$$
- *Electronvolts*: sometimes you'll see energy described in terms of electronvolts ($\text{eV}$).This is defined as *The energy gained or lost by a single electron accelerating from rest through an electric potential difference of one volt*. To use the gravitational analogy, think of it as quantifying potential energy as the energy gained / lost when you drop it from a height of 1 meter.
	- conversion: $1 \text{eV} = 1.6 \times 10^{-19} \text{J}$
	- Uses the fact that $\text{Electric potential energy} = q \times \text{Electric potential}$

**Electric potential**

Definition

$$
V = \frac{1}{4 \pi \epsilon_0} \frac{q}{r} = k \frac{q}{r}
$$
Relationship to other quantities

$$
\begin{align}
V &= \frac{U}{q} \\
\\
V &= E \cdot d
\end{align}
$$

Units
$$
\text{V} = \frac{\text{kg} \cdot \text m^2}{\text s^3 \cdot \text A} = \frac{\text J}{\text C}
$$

A note on interpreting electric potential:
- we are always interested in the difference in electric potential between two points, almost never absolute values
- since we're interested in difference, we get to decide where zero potential lives
- in the definition, zero potential is considered to be infinitely far from the charge
- protons will "fall" from a higher potential to a lower potential the same way a pen falls when you drop it
- Think of potential as a measure of "height" - it's a way of describing how much energy an object will gain / loose when it gets closer to / farther away from the charge without knowing anything about the mass
	- Gravitational potential energy is $mgh$
	- Gravitational potential is $mg$


**Resistance**
Definition

$$
R = \rho \frac{l}{A}
$$
Where $\rho$ is the resistivity, $l$ is the length of the resistor, and $A$ is the surface area.

Units are in ohms

**Current**

Definition
$$
I = \frac{Q}{t} = \frac{\text d Q}{\text d t}
$$
This is the amount of current passing over a point per unit time

**Capacitance**
Definition

$$
C = \frac{q}{V}
$$
for a parallel-plate capacitor
$$
C = \epsilon_0 \frac{A}{d}
$$
where $A$ is the surface area of one of the plates and $d$ is the distance between the two plates. If there is a material between the two plates, then switch out $\epsilon_0$ for $\kappa$, the dielectric constant, where $\kappa = \epsilon_r \cdot \epsilon_0$, where $\epsilon_r$ is the relative permittivity
This is the ability 


**Magnetic field (aka magnetic flux density)**

for a point charge:

$$
\vec B = \frac{\mu_0}{4 \pi} \frac{q \vec v \times \hat r}{r^2}
$$

near a wire:
$$
|B| = \frac{\mu_0 I}{2 \pi r}
$$
Solenoid:
$$
B = \frac{\mu_0 N I}{l}
$$

relationship to other quantities:

- Lorenz force:
$$
F = qE + q(v \times B)
$$
unit: Tesla

**Inductance**


The tendency of a conductor to oppose the change in electric current flowing through it

$$
L = \frac{\phi_B (I)}{I}
$$

Where $\phi_B$ is magnetic flux

relationship to other quantities:

Energy:
$$
U = \frac{1}{2} L I ^2
$$