import sympy
from sympy import Function, dsolve, Symbol, Q, ask
from sympy.assumptions import assuming

# symbols
t = Symbol('t', positive=True)
r = Symbol('r', positive=True)
# unknown function
u = Function('u')(t)

# assumed values
wn = 10.
u0 = 0
v0 = 0
fi = 1
tc = 2
epsilon = 0.05
zeta = 0.1

# solving ODE
f = sympy.Piecewise(
        (0, t < tc),
        (0, t > (tc+epsilon)),
        (fi, True))

ics = {u.subs(t, 0): u0,
       u.diff(t).subs(t, 0): v0,
       }
sol = dsolve(u.diff(t, t) + 2*zeta*wn*u.diff(t) + wn**2*u - f, ics=ics)
#FIXME why is this not working?

import matplotlib
matplotlib.use('TkAgg')
from sympy.plotting import plot

p1 = plot(sol.rhs, (t, 0, 5), xlabel='$t$', ylabel='$u(t)$', nb_of_points_x=10000)

