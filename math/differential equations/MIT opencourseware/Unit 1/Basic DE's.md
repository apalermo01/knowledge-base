# Basic Differential Equations
Source: https://ocw.mit.edu/courses/18-03sc-differential-equations-fall-2011/pages/unit-i-first-order-differential-equations/basic-de-and-separable-equations/


## Most important DE

$$
\dot{y} = ay \tag{1}
$$
solution:
$$
y(t) = Ce^{at}
$$
this is the exponential growth / decay model

Can solve this one without having to do anything becuase it's so ubiquotous

## Other basic examples

**1** problem: $\frac{dy}{dx} = 2x$; solution: $y(x) = x^2 + c$ (c parameterizes all solutions)

**2**: heat diffusion: Newton's law of cooling = *A body at temperature $T$ sits in an environment of temperature $T_E$*
$$ T' = -k(T-T_E) $$

**3**: Newton's law of motion (constant gravity)
$$
\frac{d^2y}{dt^2} = -g
$$
**4**: Newton's law of gravitation
$$
\frac{d^2r}{dt^2} = -GM_E / r^2
$$
**5**: simple harmonic oscillator (hooke's law)
$$
m\ddot{x} = -kx
$$
**6**: Damped Harmonic Oscillator
$$
m\ddot{x} = -kx - b\dot{x}
$$
**7**: Damped harmonic oscillator with an external force
$$
m\ddot{x} = -kx - b\dot{x} + F(t)
$$
**TODO: separation of variables**
