Sometimes just stating the limit of a function does not give a complete picture of a function's behavior - even if the limit does not exist we may still be able to say something about it's behavior. 

# One-Sided limits

Take the graph of $f(x) = \frac{|x-2|}{x-2}$:
![[Pasted image 20220908160034.png]]

Since $\lim_{x\to2} f(x)$ seems to approach different numbers from each side, we must say that the limit does not exist. However, sometimes it is useful to describe the limit as a function approaches a value from one side, aptly called a one-sided limit.

From the left:
$$
\lim_{x\to2^-} f(x) = -1
$$

and from the right:
$$
\lim_{x\to2^+} f(x) = 1
$$
Whenever I evaluate these, I like to think of left-sided limits as "coming up from the negative side" and right handed limits as "coming from the positive side".

# Infinite limits
Sometimes functions will "blow up" towards positive or negative infinity. Technically, the limit in these cases DOES NOT EXIST - however when we say the limit of a function at a certain value is infinite - we are describing the behavior of a function, NOT what value the function approaches. 

# Refs
- https://math.libretexts.org/Bookshelves/Calculus/Map%3A_Calculus__Early_Transcendentals_(Stewart)/02%3A_Limits_and_Derivatives/2.02%3A_The_Limit_of_a_Function