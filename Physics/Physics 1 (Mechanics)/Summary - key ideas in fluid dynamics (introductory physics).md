# Key ideas in fluid dynamics

## Quantities / terms

- fluid - anything that flows (generally refers to liquids / gasses)
- pressure - fluids exert a force on their containers as molecules collide with the boundaries. Refers to force per unit area (F/A)
	- hydrostatic pressure - pressure associated with fluids. increases with depth
	- thermal pressure - pressure from gasses - primarily due to random collisions with the container as a result of the gas' temperature. Constant in a container. 
	- Note: the formula for pressure is $P = \frac{F}{A}$ - the force refers to the component of force that is perpendicular to A
	- Unit: $1 \text{pascal} = 1 \text{Pa} \equiv 1 \text{N}/\text{m}^2$ 
	- If an object is submerged in a fluid - the fluid will exert an equal force in all directions
- density - measure of mass per unit volume ($\rho = m/V$)
	- units are $\text{kg}/\text{m}^3$, $\text{g}/\text{cm}^3$ is  widly used too. Conversion: $1 \text{g}/\text{cm}^3 = 1000 \text{kg}/\text{m}^ 3$
	- 
- buoyancy
- ideal fluid

## Hydrostatic pressure
$$
p = p_0 + \rho g d 
$$
Describes the pressure in a liquid at depth $d$, where $p_0$ is the pressure at the surface (1atm if at sea level) and $\rho$ is the density of the liquid. 

## Archimedes principle

Bouyant force = weight of displaced fluid

$$
F_g = \rho_{fluid} V_{displaced} g 
$$
For a partially submerged object:
$$
\text{fraction submerged} = f = \frac{V_{fluid}}{V_{obj}} = \frac{\rho_{obj}}{\rho_{fluid}} 
$$
## Continuity equation
The amount of fluid passing one point must be the same as the amount of fluid flowing past another point

$$
Q_1 = Q_2 \rightarrow \rho_1 A_1v_1 = \rho_2 A_2v_2 \rightarrow (\text{constant density}) A_1v_1 = A_2v_2
$$


## Bernoulli's principle
$$
P + \frac{1}{2}\rho v^2 + \rho g h = \text{constant}
$$
