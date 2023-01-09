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

