## Quantities
**Impedance** - analogue of resistance in DC circuits. Associated with a phases angle -> becuase peaks of current and voltage do not occur at the same time

**Average Power** - accounts for phase difference in AC circuits

**angular frequency ($\omega$)** - relates to angular frequency of sin wave representing emf

## Formulas

**Ohm's law for AC circuits:**
$$
I = \frac{V}{Z}
$$
$I$ and $V$ correspond to rms / effective values

**Average power:**
$$
P_\text{avg} = VI \cos\phi = V_\text{rms} I_\text{rms}
$$
Where $\phi$ is the phase angle

**Peak voltage and current**

$$
\begin{align}
V &= \frac{V_m}{\sqrt{2}} \\
I &= \frac{I_m}{\sqrt{2}}
\end{align}
$$

## Impedance of various components

**Resistor**
- same as DC resistance (use rms values for V and I)

**Capacitor**
- called capacitave reactance
- complex contribution to impedance ($\frac{-j}{\omega c}$)
$$
X_c = \frac{1}{\omega C}
$$
- pure C circuit will cause voltage to LAG current by 90 deg

**Inductor**
- called inductive reactance
- complex contribution to impedance: $j\omega L$ 
$$
X_L = \omega L
$$
- pure L circuit will cause voltage to LEAD current by 90 deg

**RC circuit**
- voltage LAGS current, but by less than 90 deg

$$
\begin{align}
\text{contribution to complex impedance} &= R - \frac{j}{\omega C} \\
Z &= \sqrt{R^2 + (\frac{1}{\omega C})^2} \\
\text{phase angle} = \phi &= \tan^{-1} \frac{-1/\omega C}{R}
\end{align}
$$
(note: formula for phase angle aligns with opp / adj = (impedance from capacitor) / (impedance from resistor))

**RL circuit**
- voltage LEADS current, but by less than 90 deg
$$
\begin{align}
\text{contribution to complex impedance} &= R + j \omega L \\
Z &= \sqrt{R^2 + \omega^2 L^2} \\
\text{phase angle} = \phi &= \tan^{-1} \frac{\omega L}{R}
\end{align}
$$

**RLC Series Circuit**

example of a resonant circuit. Minimum impedance of $Z=R$ at resonant frequency

Resonant condition:
$$
\begin{align}
Z &= R \\
\omega &= \frac{1}{\sqrt{LC}} \\
X_C &= L_C \\
\text{Phase} = \phi &= 0
\end{align}
$$

Normal conditions:
$$
\begin{align}
X_C &= \frac{1}{\omega C} \\
X_L &= \omega L \\
Z &= \sqrt{R^2 + (X_L - X_C)^2} \\
\text{phase} = \phi &= \tan^{-1} \frac{X_L - X_C}{R}
\end{align}
$$

**RLC Parallel (and combination)**
Dealing with parallel circuits can be more complex and tedious becuase each branch can have a different phase angle that combines differently. 

In general, impedance combines like resistance:

$$
Z_\text{parallel} = (\frac{1}{Z_1} + \frac{1}{Z_2} + ...)^2
$$
$$
Z_\text{series} = Z_1 + Z_2 + ...
$$

remember that we're doing these calculations with complex numbers, so the denominators must be rationalized. 

See [this](http://hyperphysics.phy-astr.gsu.edu/hbase/electric/impcom.html#c1) for a more detailed overview of the forms we can use for dealing with complex impedance