# Radioactive decay

## General equation
The decay rate of a substance is proportional to the amout of the original substance, leading to the differential equation:

$$
-\frac{dN}{dt} = \lambda N
$$
Solving this equation gives the solution:
$$
N = N_0 e^{-\lambda t}
$$
where:
- $N_0$ = original amount of substance
- $N$ = amount of substance after time $t$
- $t$ = time
- $\lambda$ = decay constant



**half life** - amount of time that needs to pass for half of the original substance to remain


to find half life- solve this equation for $T_{1/2}$
$$
\frac{N_0}{2} = N_0 e^{-\lambda T_{1/2}}
$$
solution: 
$$
T_{1/2} = \frac{0.693}{\lambda}
$$
**if you know the half life of a decay, you can find** $\lambda$ 


**Activity** - magnitude of the decay rate
$$
A = -\frac{dN}{dt} = \lambda N = \lambda N_0 e^{-\lambda t}
$$

## Types of radioactive decay
source: table 17.3.1 of [2]

 | particle | description | symbol | mass | penatrating power | ionizing power | shielding |
 | --- | --- | --- | --- | --- | --- | --- |
 | Alpha | helium nucleus | $\alpha$ | 4 amu | very low | very high | paper / skin
 | Beta | electron | $\beta$ | 1/2000 amu | intermediate | intermediate | aluminum
 | Gamma | high energy photon | $\gamma$ | energy only | very high | very low | 2 inches lead
**two kinds of beta decay**
- beta-plus: emits a positron - new atom has fewer protons than previous atom
- beta-minus: emits an electron - new atom has one more proton than previous atom

electron capture [TODO]

## Refs
- [ 1] https://phys.libretexts.org/Bookshelves/University_Physics/Book%3A_University_Physics_(OpenStax)/University_Physics_III_-_Optics_and_Modern_Physics_(OpenStax)/10%3A__Nuclear_Physics/10.04%3A_Radioactive_Decay
- [ 2] https://chem.libretexts.org/Courses/can/intro/17%3A_Radioactivity_and_Nuclear_Chemistry/17.03%3A_Types_of_Radioactivity%3A_Alpha_Beta_and_Gamma_Decay
- [ 3] https://education.jlab.org/glossary/betadecay.html
- [ 4] https://www.radioactivity.eu.com/site/pages/Electron_Capture.htm