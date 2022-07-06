# Solving quadratic equations

There are 4 ways to solve any quadratic equation:

- Graphing
- Factoring
- Quadratic Formula
- Completing the Square

Note: there are several terms used to talk about the solutions to quadratic equations: *solutions*, *zeros*, *roots*

There are 3 types of solutions to quadratic equations:

- 2 solutions
- 1 solution
- no solution

By the fundamental theorem of algebra, any polynomial of degree n has exactly n solutions, so how can we get no solution for a quadratic? The answer is that if you have no real roots, then then solutions are complex. Similarly, If you have 1 solution / root, then we call that a repeated root.  

# 1. Solving quadratic equations by graphing

To solve a quadratic equation by graphing, look for where the graph crosses the x-axis. The x-coordinates where the graph meets the axis are the roots


```python
import matplotlib.pyplot as plt
import numpy as np

def plot_poly(ax, coefs, title):

    ### https://stackoverflow.com/a/31558968
    # Move left y-axis and bottim x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    
    ### find solutions to quadratic
    roots = np.roots(coefs)
    x = np.linspace(-10, 10, 1000)
    y = np.polyval(coefs, x)

    ### make plot
    ax.plot(x, y)
    if np.isreal(roots).all():
        ax.scatter(roots, np.polyval(coefs, roots), c='r')
    ax.set_xlim([-10, 10])
    ax.set_ylim([-10, 10])
    ax.set_title(title)

fig, ax = plt.subplots(ncols=3, figsize=[12, 3])
plot_poly(ax[0], (0.5, -4, -2), "2 real roots")
plot_poly(ax[1], (1, -10, 25), "1 repeated root")
plot_poly(ax[2], (1, -2, 5), "no real roots")
```


    
![png](output_2_0.png)
    


In the above plots, the red dots represent the roots. 

If there are 2 real roots, the the graph crosses the x-axis twice. If there is a repeated root, the graph appears to "kiss" the plot. If there are no real roots, the graph will not touch the x-axis

# 2. Solving by factoring

Start with a quadratic equation in standard form:

$$
ax^2 + bx + c = 0
$$

Find 2 numbers that multiply to the same product as $ac$ and add to $b$. Let's call these numbers $m$ and $n$. If no such numbers exist then factoring will not work - either the roots are irrational or there are no real roots. 

Split the middle term of the expression

$$
ax^2 + mx + nx + c = 0
$$


Now factor each pair of terms. You should get something that looks like this:

$$
dx(fx-g) + e(fx-g) = 0
$$

where $d$, $e$, $f$, and $g$ are some constants

Notice that both these terms have have $(fx-g)$, so we can factor that whole thing out and get

$$
(dx+e)(fx-g) = 0
$$

Of course, based on how things play out the signs may look different. 

Notice that we have 0 on the right side and a product on the left side. The only way a product of any two numbers is zero is if one of the numbers in the product is zero, which means:

$$
dx+e = 0
$$

or 

$$
fx-g = 0
$$

Each solution from these equations is a root of the quadratic. If both terms are the same (i.e. $(dx+e) = (fx-g)$), then we have a repeated root.

# 3. Solving by the quadratic formula

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

$b^2 - 4ac$ is called the discriminate and tells you what kind of solutions you will get

$$
b^2 - 4ac > 0  \quad \text{2 real solutions}
$$
$$
b^2 - 4ac = 0 \quad \text{2 repeated solutions}
$$
$$
b^2 - 4ac < 0 \quad \text{no real solutions (both solutions will be complex)}
$$

# 4. Solving by completing the square

reference: https://www.mathsisfun.com/algebra/completing-square.html

Start by dividing both sides of the equation by a, so you have an equation in the form of:
$$
a\big[x^2 + bx + c\big] = 0
$$

To complete the square, add and subtract $(b/2)^2$ from the left side

$$
a\big[x^2 + bx + c + (\frac{b}{2})^2 - (\frac{b}{2})^2\big] = 0
$$

regroup

$$
a\big[x^2 + bx + (\frac{b}{2})^2 - (\frac{b}{2})^2 + c\big] = 0
$$

The first 3 terms ($x^2 + bx + (\frac{b}{2})^2)$) form a perfect square, so we can easily factor it. After some re-arranging we'll wind up with something that looks like

$$
a(x+d)^2 + e = 0
$$

This would be the final answer if we were just completing the square

For a shortcut, we could jump to:

$$
d = \frac{b}{2a}
$$
$$
e = c - \frac{b^2}{4a}
$$

To solve, we finish it off like a normal equation, starting with

$$
(x+d)^2 = -\frac{e}{a}
$$

If we were to keep going, and replace $e$ and $d$ with the expressions above, we would get the quadratic formula back.

Note: this form, after completing the square is called **vertex form**, more commonly written as
$$
a(x-h)^2 + k = 0
$$

where the vertex of the parabola is $(h, k)$

Note: I like to link up this act of adding and subtracting $(\frac{b}{2})^2$ to multilpying and dividing by the same number when manipulating fractions - both are examples of doing essentially nothing to make manipulating the expression easier. 
