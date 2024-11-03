In my experience, many students first learning about waves feel like they're being thrown a ton of equations and have trouble organizing and making sense of everything, so here I'm going to break down the key equations / formulas / properties 

#todo: sketch of a wave with all the fundamental properties

**What is a wave?**: A wave is used to describe any phenomenon where some quantity exhibits a cyclic / repetitive behavior (e.g. the same thing happening at fixed intervals of time). Here are a few examples:
- A pocket of air compressing / expanding 440 times per second is perceived as the A above middle C
- The height of a fixed point in a pond oscillates at some fixed frequency is a water wave


**The equation for a wave:**
In simple terms, the properties of any wave can be written down like this when considering both position and time:
$$
y(x, t) = \sin(kx - \omega t + \phi)
$$ 
and like this when position is fixed:
$$
x(t) = \sin(\omega t + \phi)
$$
where $k$ is called the *wave number*, $\omega$ is the *angular frequency* and $\phi$ is the phase.

More generally, you're dealing with a wave any time a physical system can be described by this differential equation (**don't worry about this until you're in classical mechanics**)

$$
\frac{\partial^2 u}{\partial t ^2} = c^2 \nabla^2 u
$$


**Key properties of a wave**

- **Period**: The period ($T$) of a wave is how long it takes to complete one full cycle. It is the inverse of frequency: $T = 1/f$. Unit = [s] (second).
- **Frequency**: The frequency ($f$) of a wave is how many cycles occur in a given period. It is the inverse of period: $f = 1/T$. Unit = [Hz] (Hertz) where $1 \text{Hz} = 1 / \text{s}$ or $s^{-1}$.  
- **Angular Frequency**: The angular frequency of a wave $\omega$ is how many radians the process goes through in a given period. It's related to frequency by $\omega = 2 \pi f$. 

## References
- http://hyperphysics.phy-astr.gsu.edu/hbase/Sound/wavplt.html
- http://hyperphysics.phy-astr.gsu.edu/hbase/shm.html