#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
from matplotlib import animation, rc
rc('animation', html='jshtml')
from scipy.optimize import fsolve
from scipy.optimize import curve_fit
from scipy.linalg import norm
from scipy.integrate import solve_ivp
from scipy.integrate import quad
from scipy.integrate import odeint
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('pdf', 'svg')
from pathlib import Path
fn = Path('/Users/madsmikkelsen/Desktop/SIR.pdf').expanduser()

plt.rcParams['figure.figsize'] = (15,10)
plt.rc("axes", labelsize=18)
plt.rc("xtick", labelsize=16, top=True, direction="in")
plt.rc("ytick", labelsize=16, right=True, direction="in")
plt.rc("axes", titlesize=22)
plt.rc("legend", fontsize=16, loc="upper left")
plt.rc("figure", figsize=(10, 7))

sns.set_style("darkgrid")
sns.set_context("paper")
sns.set(font_scale=1.4)
mycolors = ['#C188F7','#F79288','#7FB806','#F59B18']
sns.set_palette("Set2") 


# In[8]:


fig, ax = plt.subplots()

# Population
N = 1000

# Start værdier
I0 = 1
R0 = 0
S0 = N-I0-R0

# Risiko for smitte
beta = 0.5

# Helbredelsesrate (1/10 dage)
gamma = 1/10

# tidsinterval
t = np.linspace(0, 60, 200)

# SIR ODEs
def derive(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Startbetingelser
y0 = S0, I0, R0

# Løsning af ODE
løs = odeint(derive, y0, t, args=(N, beta, gamma))
S, I, R = løs.T

# Plot
ax.plot(t, S/N, label='Susceptible')
ax.plot(t, I/N, label='Infected')
ax.plot(t, R/N, label='Recovered')

# Aksetitler
ax.set_xlabel('Time [Days]')
ax.set_ylabel('Population [pr. 1000]')
ax.set_title('SIR')

# Legend
ax.legend(loc='best')

disp = False
plt.savefig(fn, bbox_inches='tight')


# In[ ]:




