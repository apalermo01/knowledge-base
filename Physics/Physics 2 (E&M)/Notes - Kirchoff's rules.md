# Kirchoff's Rules

These are a set of rules to use when analyzing complex / branching circuits. e.g.:
![[circuit problem.png]]
(source: 3000 solved problems in physics fig 27-40 (pg 462))

## Junction rule
This rule deals with the current at nodes / branches. Think of the current as the flow of water through a system of rivers. When a series of rivers meet at a junction, the volume of water that goes into the junction is the same as the amount of water that comes out. Same thing with electrical current:

$$
\sum_{\text{current flowing in}} I = \sum_{\text{current flowing out}} I
$$
Most of the time it's not clear whether current is flowing in or out of a given branch of a junction, so many times the equation looks like this:
$$
\sum_{\text{all branches}} I = 0
$$
where $I$ is positive if current is flowing in and negative if current is flowing out.

## Loop rule
This rules deals with voltage drops around loops in the circuit. Think of these as a roller coaster. Batteries are like chain lifts that lift the train up to a higher potential, and components / resistors are like the drops- bringing the train from a higher gravitational potential to a lower gravitational potential. No matter what, the train finishes at the same elevation that it started at (in other words, the net change in gravitational potential is zero), resulting in: 

$$
\sum_{loop} V = 0
$$

## problem solving with kirchoff's loop rules
 - each loop / junction will give you an equation
 - When using the loop or junction rule, just take a guess as to which way the currents are flowing. If you're wrong about the direction, you'll know becuase current will be negative
 - If there are $n$ components you need to solve for, you will need at least $n$ equations to solve for all of them. 

refs:
- https://phys.libretexts.org/Bookshelves/University_Physics/Book%3A_Calculus-Based_Physics_(Schnick)/Volume_B%3A_Electricity_Magnetism_and_Optics/B12%3A_Kirchhoffs_Rules_Terminal_Voltage