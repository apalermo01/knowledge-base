
# Basics
$$
\lim_{x\to a} x = a
$$
$$
\lim_{x \to a} c = c
$$

$f(x)$ and $g(x)$ are defined for all $x \neq a$ on some open interval containing a. $\lim_{x \to a} f(x) = L$ and $\lim_{x \to a} g(x) = M$. $c$ is a constant. 

sum:
$$
\lim_{x\to a} (f(x) + g(x)) = \lim_{x \to a} f(x) + \lim_{x \to a} g(x) = L + M
$$
difference: 
$$
\lim_{x\to a} (f(x) - g(x)) = \lim_{x \to a} f(x) - \lim_{x \to a} g(x) = L - M
$$

constant multiple:
$$
\lim_{x \to a} cf(x) = c \cdot \lim_{x \to a} f(x) = cL
$$
product:
$$
\lim_{x \to a} (f(x) \cdot g(x)) = \lim_{x \to a} f(x) \cdot \lim_{x \to a} g(x) = L \cdot M
$$
quotient:
$$
\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{{\lim_{x \to a} f(x)}}{{\lim_{x \to a} g(x)}} = \frac{L}{M}
$$
power: 
$$
\lim_{x \to a} (f(x))^n = (\lim_{x \to a} f(x))^n = L^n
$$
root: 
$$
\lim_{x \to a} \sqrt[n]{f(x)} = \sqrt[n]{\lim_{x \to a} f(x)} = \sqrt[n]{L}

$$

$p(x)$ and $q(x)$ are polynomial functions, $a$ is a real number
$$
\lim_{x \to a} p(x) = p(a)
$$
$$
\lim_{x \to a} \frac{p(x)}{q(x)} = \frac{p(a)}{q(a)}
$$

# Polynomials

polynomial and rational functions

if $p(x)$ and $q(x)$ are polynomial functions and $a$ is a real number, then:
$$
\lim_{x\to a} \frac{p(x)}{q(x)} = \frac{p(a)}{q(a)} 
$$
If $p(a)/q(a)$ results in the indeterminate form $0/0$:
- try factoring both functions and cancel any terms
- if one function has a difference of a square root, multiply the top and bottom by the congugate
- If there is a complex fraction, simplify it

# Squeeze theorem
Very useful for finding the limits of trig functions

Let $f(x), g(x), \text{and} h(x)$ be functions defined for all $x \neq a$. If
$$
f(x) \leq g(x) \leq h(x)
$$
and 
$$
\lim_{x\to a} f(x) = L = \lim_{x\to a} h(x)
$$
then

$$
\lim_{x\to a} g(x) = L
$$
See the second reference (Stewart section 2.3) for examples

# Refs
[1] https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Calculus_(OpenStax)/02%3A_Limits/2.3%3A_The_Limit_Laws
[2] https://math.libretexts.org/Bookshelves/Calculus/Map%3A_Calculus__Early_Transcendentals_(Stewart)/02%3A_Limits_and_Derivatives/2.03%3A_Calculating_Limits_Using_the_Limit_Laws