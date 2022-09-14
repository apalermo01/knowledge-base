# Intro to limits


**A limit describes the behavior of a function AROUND a point, as opposed to what the function does AT the point**

**Another definition: as $x$ gets closer to $a$, $f(x)$ gets closer and stays close to $L$ 

For example, what happens to the function $f(x) = x^2$ around the point $x=2$?

First, let's see what happens when we approach from the left:
| $x$ | $f(x)$ |
| --  | -- |
| 1.5  | 2.25 |
| 1.9 | 3.61 |
| 1.99 | 3.9601 |
| 1.999 | 3.996001 |
| 1.9999 | 3.99960001 |

Now, let's see what happens when we approach from the right:

| $x$ | $f(x)$ |
| --  | -- |
| 2.5  | 6.25 |
| 2.1 | 4.41 |
| 2.01 | 4.0401 |
| 2.001 | 4.004001 |
| 2.0001 | 4.00040001 |

When we let x approach 2 from each side, the output of the function gets closer and closer to 4, therefore we can say that the limit as x approaches 2 of $x^2$ is 4, written as:


$$
\lim_{x\to2} x^2 =  4
$$
# How to tell if a limit exists

**Here are conditions under which a limit will now exsits**


### Jump discontinuity

For a limit to exist, we must have $\lim_{x \to a^+} f(x) = \lim_{x \to a^-} f(x)$
![[Pasted image 20220827144517.png]]

#### Vertical asymptotes
![[Pasted image 20220827144609.png]]

#### violent oscillations
![[Pasted image 20220827144827.png]]


REMEMBER: **IF THE LIMIT GOES TO INFINITY, THEN THE LIMIT DOES NOT EXIST**

# Refs
https://socratic.org/calculus/limits/determining-when-a-limit-does-not-exist
https://www.khanacademy.org/math/ap-calculus-ab/ab-limits-new/ab-1-2/a/limits-intro