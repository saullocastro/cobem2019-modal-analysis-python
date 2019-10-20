import numpy as np

m = 3.
k = 100.
wn = np.sqrt(k/m)
u0 = 0
v0 = 0
tn = 2
zeta = 0.1
fi = 7
dt = 0.05
wd = wn*(1 - zeta**2)**0.5

def h(t, tn):
    return 1/(m*wd)*np.heaviside(t-tn, 1)*np.exp(-zeta*wn*(t-tn))*np.sin(wd*(t-tn))

def u(t, tn):
    return fi*dt*h(t, tn)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

t = np.linspace(0, 10, 1000)
plt.plot(t, u(t, tn))
plt.xlabel('$t$')
plt.ylabel('$u(t)$')
plt.show()

