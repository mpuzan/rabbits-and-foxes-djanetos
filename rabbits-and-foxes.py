uman
# coding: utf-8

# # Rabbits and foxes
# 
# There are initially 400 rabbits and 200 foxes on a farm (but it could be two cell types in a 96 well plate or something, if you prefer bio-engineering analogies). Plot the concentration of foxes and rabbits as a function of time for a period of up to 600 days. The predator-prey relationships are given by the following set of coupled ordinary differential equations:
# 
# \begin{align}
# \frac{dR}{dt} &= k_1 R - k_2 R F \tag{1}\\
# \frac{dF}{dt} &= k_3 R F - k_4 F \tag{2}\\
# \end{align}
# 
# * Constant for growth of rabbits $k_1 = 0.015$ day<sup>-1</sup>
# * Constant for death of rabbits being eaten by foxes $k_2 = 0.00004$ day<sup>-1</sup> foxes<sup>-1</sup>
# * Constant for growth of foxes after eating rabbits $k_3 = 0.0004$ day<sup>-1</sup> rabbits<sup>-1</sup>
# * Constant for death of foxes $k_1 = 0.04$ day<sup>-1</sup>
# 
# Also plot the number of foxes versus the number of rabbits.
# 
# Then try also with 
# * $k_3 = 0.00004$ day<sup>-1</sup> rabbits<sup>-1</sup>
# * $t_{final} = 800$ days
# 
# *This problem is based on one from Chapter 1 of H. Scott Fogler's textbook "Essentials of Chemical Reaction Engineering".*
# 

# # Solving ODEs
# 
# *Much of the following content reused under Creative Commons Attribution license CC-BY 4.0, code under MIT license (c)2014 L.A. Barba, G.F. Forsyth. Partly based on David Ketcheson's pendulum lesson, also under CC-BY. https://github.com/numerical-mooc/numerical-mooc*
# 
# Let's step back for a moment. Suppose we have a first-order ODE $u'=f(u)$. You know that if we were to integrate this, there would be an arbitrary constant of integration. To find its value, we do need to know one point on the curve $(t, u)$. When the derivative in the ODE is with respect to time, we call that point the _initial value_ and write something like this:
# 
# $$u(t=0)=u_0$$
# 
# In the case of a second-order ODE, we already saw how to write it as a system of first-order ODEs, and we would need an initial value for each equation: two conditions are needed to determine our constants of integration. The same applies for higher-order ODEs: if it is of order $n$, we can write it as $n$ first-order equations, and we need $n$ known values. If we have that data, we call the problem an _initial value problem_.
# 
# Remember the definition of a derivative? The derivative represents the slope of the tangent at a point of the curve $u=u(t)$, and the definition of the derivative $u'$ for a function is:
# 
# $$u'(t) = \lim_{\Delta t\rightarrow 0} \frac{u(t+\Delta t)-u(t)}{\Delta t}$$
# 
# If the step $\Delta t$ is already very small, we can _approximate_ the derivative by dropping the limit. We can write:
# 
# $$\begin{equation}
# u(t+\Delta t) \approx u(t) + u'(t) \Delta t
# \end{equation}$$
# 
# With this equation, and because we know $u'(t)=f(u)$, if we have an initial value, we can step by $\Delta t$ and find the value of $u(t+\Delta t)$, then we can take this value, and find $u(t+2\Delta t)$, and so on: we say that we _step in time_, numerically finding the solution $u(t)$ for a range of values: $t_1, t_2, t_3 \cdots$, each separated by $\Delta t$. The numerical solution of the ODE is simply the table of values $t_i, u_i$ that results from this process.
# 

# # Euler's method
# *Also known as "Simple Euler" or sometimes "Simple Error".*
# 
# The approximate solution at time $t_n$ is $u_n$, and the numerical solution of the differential equation consists of computing a sequence of approximate solutions by the following formula, based on Equation (10):
# 
# $$u_{n+1} = u_n + \Delta t \,f(u_n).$$
# 
# This formula is called **Euler's method**.
# 
# For the equations of the rabbits and foxes, Euler's method gives the following algorithm that we need to implement in code:
# 
# \begin{align}
# R_{n+1} & = R_n + \Delta t \left(k_1 R_n - k_2 R_n F_n \right) \\
# F_{n+1} & = F_n + \Delta t \left( k_3 R_n F-n - k_4 F_n \right).
# \end{align}

# In[5]:

get_ipython().magic('matplotlib inline')
import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate


# In[ ]:




# In[2]:

k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04
end_time = 600.
step_size = 1.
times = np.arange(0, end_time, step_size)
rabbits = np.zeros_like(times)
foxes = np.zeros_like(times)
rabbits[0] = 400.
foxes[0] = 200.
for n in range(len(times)-1):
    delta_t = times[n+1] - times[n]
    rabbits[n+1] = rabbits[n] + delta_t * (k1 * rabbits[n] - k2 * rabbits[n] * foxes[n])
    foxes[n+1] = foxes[n] + delta_t * (k3 * rabbits[n] * foxes[n] - k4 * foxes[n])


# In[3]:

plt.plot(times, rabbits, label='rabbits')
plt.plot(times, foxes, label='foxes')
plt.legend(loc="best") # put the legend at the best location to avoid overlapping things
plt.show()


# In[4]:

k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04
def solve_by_euler(step_size = 1.):
    """
    Evaluate by simple Euler, with the given step size.
    
    Returns the peak number of foxes.
    """
    end_time = 600.
    times = np.arange(0, end_time, step_size)
    rabbits = np.zeros_like(times)
    foxes = np.zeros_like(times)
    rabbits[0] = 400
    foxes[0] = 200
    for n in range(len(times)-1):
        delta_t = times[n+1] - times[n]
        rabbits[n+1] = rabbits[n] + delta_t * (k1 * rabbits[n] - k2 * rabbits[n] * foxes[n])
        foxes[n+1] = foxes[n] + delta_t * (k3 * rabbits[n] * foxes[n] - k4 * foxes[n])
    return foxes.max()

step_sizes = []
maximums = []
for i in range(20):
    print(i)
    step_size = 10**(1-i/5)
    print("step size",step_size)
    maximum = solve_by_euler(step_size)
    print("max foxes",maximum)
    step_sizes.append(step_size)
    maximums.append(maximum)


# Trying to use scipy.integrate.odeint

# In[15]:

import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
end_time = 600.
step_size = 1.
times = np.arange(0, end_time, step_size)
r0 = 400
r=400
f0 = 200
f=200
y0=[r0,f0]
k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04
def rabbits_and_foxes(y0, times):
    drdt=[k1*r-k2*r*f, k3*r*f-k4*f]
    return drdt
solr=scipy.integrate.odeint(rabbits_and_foxes, y0, times)
plt.plot(times, solr)
plt.show()


# In[1]:

import scipy.integrate
help(scipy.integrate.odeint)


# Kinetic Monte Carlo method

# In[35]:

import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import scipy.integrate
import math
t=0
step_size=1
end_time=600
times = np.arange(0, end_time, step_size)
rabbits=np.zeros_like(times)
rabbits[0]=400
foxes=np.zeros_like(times)
foxes[0]=200
k1 = 0.015
k2 = 0.00004
k3 = 0.0004
k4 = 0.04
while t<end_time-1:
    rabbit_birth=k1*rabbits[t]
    rabbit_death=k2*rabbits[t]*foxes[t]
    fox_birth=k3*foxes[t]*rabbits[t]
    fox_death=k4*foxes[t]
    rchange=rabbit_birth-rabbit_death
    fchange=fox_birth-fox_death
    u0=random.random()
    rabbits[t+1]=rabbits[t]+u0*rchange
    foxes[t+1]=foxes[t]+u0*fchange
    t=t+1
plt.plot(times,rabbits,times,foxes)
plt.show()


# So far, in the times I've tried this algorithm, I have never obtained a "second peak" - the foxes have always died out completely by around 300 days, this being the case after about 20 simulations. Therefore, in my (likely flawed) script, there is a 100% chance that the foxes will die out by 600 days. I could not answer the first two questions.
# I should say that through this assignment, the most important thing I've learned is that I really need to return to my old notes on differential equations, as I feel many of the errors I encountered working with scipy.integrate.odeint came from my failure to remember how to work with ODEs. I guess I've also learned the basics of loops and working with Git (the fact that I got this loop working at all, and have been consistently able to save it to Git, is a victory in my mind).

# In[ ]:



