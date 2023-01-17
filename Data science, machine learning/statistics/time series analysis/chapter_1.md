# Chapter 1 on Time Series Analysis and It's Applications

*time series* can be defined as a collection of random variables indexed according to the order they are obtained in time. 

A collection of variables ${x_t}$, indexed by $t$, is called a **stochastic process**

The observed values are called the **realization** of the stochstic process. 

**aliasing** - the phenomenon where the appearence of a time series dataset can look completely different due to a bad sampling rate. For example, the wheels on a car in a movie can appear to be turning backward if the frame rate of the camera is insufficient. 

The key differentiator between many time series datasets is the degree of smoothness. One aexplanation for the smoothness is that adjacent points are *correlated* - so that the value at index $t$ is depends on the value at $t-1, t-2, ...$

## Some models

**White noise**

We can model a generated series as a collection of uncorrelated random variables, $w_t$ with mean 0 and finite variance $\sigma_w^2$ -> $w_t \sim \text{wn} (0, \sigma_w^2)$

we can require the noise be *independent and identically distributed*: $w_t \sim \text{iid} (0, \sigma_w^2)$

can also specify that the noise is gaussian: $w_t \sim \text{iid N}(0, \sigma_w^2)$

If the stochastic behavior of a process can be described by a white noise model, then classical methods would suffice (since that would statisfy the assumption that all the samples are independent). 

**moving averages and filtering**

we can filter out rapid oscillations and enhance slower oscillations by taking a moving average. ex:
$$
v_t = \frac{1}{3}(w_{t-1} + w_t + w_{t+1})
$$

**autogregressions**

regression / prediction of the current value based on the past 2 values of the series (so this would be a second-order equation):

$$
x_t = x_{t-1} - .9x_{t-2} + w_t
$$

**random walk with drift**

$$
x_t = \delta + x_{t-1} + w_t
$$

initial condition is $x_0 = 0$, $w_t$ is some white noise, and the constant $\delta$ is called *drift*. When $\delta = 0$, this is a random walk

we can also rewrite it as a cumulative sum of white noise variants:

$$
x_t = \delta t + \sum_{j=1}^{t} w_j
$$

**signal in noise**

many models use some underlying signal with constant period plus noise

$$
x_t = 2 \cos{\omega t} + w_t
$$

*signal to noise ratio* = $\text{amplitude of signal} / \sigma_w$ (or some function of this)

**simple addative models**

$$
x_t = s_t + v_t
$$

$s_t$ is some underlying signal and $v_t$ is a time series that could be white noise or correlated over time. 


## Measures of Dependence
A complete description of a time series can be providedd by a joint distribution function, evaluated as the probability that the values o fthe series are jointly less than the n constants $c_1, c_2, ...$

$$
F_{t_1, t_2, ..., t_n}(c_1, c_2, ..., c_n) = \text{Pr}(x_{t_1} \leq c_1, x_{t_2} \leq c2, ..., x_{t_n} \leq c_n)
$$

Usually, these distribution fucntions can't be written unless the random variables are jointly normal. 

**marginal distribution function** - the probability distribution of one term in the probability distribution of the full vector.

$$
F_t(x) = P\{x_t \leq x\}
$$
**marginal density function** - often good for examining marginal behavior of a series

$$
f_t(x) = \frac{\partial F_t (x) }{\partial x}
$$

another informative marginal description: mean function

**mean function**

$$
\mu_{xt} = \text{E}(x_t) = \int_{-\infty}^{\infty} x f_t(x) dx
$$


ex: mean of white noise is 0

ex: mean of a random walk with drift is $/delta t$

lack of independence between two adjacent values can be adressed numerically:

**autocovariance function**

$$
\gamma_x (s, t) = \text{cov}(x_s, x_t) = \text{E}[(x_s - \mu_s)(x_t - \mu_t)] 
$$

**covariance of linear combinations** (good for finding the covariance of filtered series)

if r.v.'s 
$$
U = \sum_{j=1}^m a_j X_j \text{and} V = \sum_{k=1}^r b_k Y_k
$$

are linear combinations of random variables $\{X_j\}$ and $\{Y_k\}$ then

$$
\text{cov} (U, V) = \sum_{j=k}^m \sum_{k=1}^r a_j b_k \text{cov}(X_j, Y_k)
$$


**autocorrelation function (ACF)**

$$
\rho(s, t) = \frac{\gamma(s, t)}{\sqrt{\gamma(s, s) \gamma(t, t)}}
$$

measures the linear predictability of the time series at time $t$ using only the value $x_s$. 

**cross-covariance function**
$$
\gamma_{xy} (s, t) = \text{cov}(x_s, y_t) = \text{E}[(x_s - \mu_{xs})(y_t - \mu_{yt})]
$$

**cross-correlation function**
$$
\rho_{xy} (x, t) = \frac{\gamma_{xy}(s, t)}{\sqrt{\gamma_x (s, s) \gamma_y (t, t)}}
$$


# 1.4 Stationary Time Series

**Strictly stationary** - a time series in which the probabalistic behavior of every collection of values $\{x_{t_1}, x_{t_2}, ..., x_{t_k}\}$ is identical to that of the time shifted set $\{x_{t_1 + h}, x_{t_2 + h}, ..., x_{t_k + h}\}$

In other words:

$$
\text{Pr} \{x_{t_1}  \leq c_1, ... x_{t_k} \leq c_k\} = \text{Pr}\{x_{t_1 + h} \leq c_1, ... x_{t_k + h}
$$


for $k=1$

$$
\text{Pr} \{x_s \leq c\} = \text{Pr} \{ x_t \leq c \}
$$

As an example: this implies that the probability that the values of a time series sampled hourly is negative at 1 am is the same as at 10 am.

If the mean function $\mu_t$ exists, then this implies that $\mu_s = \mu_t$ -> therefore $\mu_t$ is constant

Note: random walk with drift is not strictly stationary becuase the mean function changes with time.

for $k=2$:

$$
\text{Pr} \{ x_s \leq c_1, x_t \leq c_2 \} = \text{Pr} \{ x_{s+h} \leq c_1, x_{t+h} \leq c_2 \}
$$

If the variance function of the process exists, then the last 2 equations imply that the autocovariance is:

$$
\gamma (s, t) = \gamma (s+h, t+h)
$$

Essentially, the autocovariance depends only on the time difference between s and t, not on the actual times

**weakly stationary**


**autocorrelation function (ACF) of a stationary time series**

**trend stationary**


**cross-correlation function (CCF)**

**prediction using cross-correlation**

**linear process**

**gaussian process**


