import sympy
from sympy import Function, dsolve, Symbol, Q, ask
from sympy.assumptions import assuming

# symbols
t = Symbol('t', positive=True)
r = Symbol('r', positive=True)
#zeta = 0.2
zeta = Symbol('\zeta', positive=True)
wn = Symbol('\omega_n', positive=True)
epsilon = Symbol(r'\varepsilon', positive=True)
tc = Symbol('t_c', positive=True)
fi = Symbol('f_i', positive=True)
u0 = Symbol('u_0')
v0 = Symbol('v_0')

# unknown function
u = Function('u')(t)

# solving ODE
f = sympy.Piecewise(
        (0, t < tc),
        (0, t > (tc+epsilon)),
        (fi, True))
ics = {u.subs(t, 0): u0,
       u.diff(t).subs(t, 0): v0,
       }
sol = dsolve(u.diff(t, t) + 2*zeta*wn*u.diff(t) + wn**2*u - f, ics=ics)

import matplotlib
matplotlib.use('TkAgg')
from sympy.plotting import plot

res = sol.rhs.subs({
wn: 10.,
u0: 0,
v0: 0,
fi: 1,
tc: 2,
epsilon: 0.05,
zeta: 0.9})
#FIXME why is this not working? Generating complex number at the end
p1 = plot(res, (t, 0, 5), xlabel='$t$', ylabel='$u(t)$', nb_of_points_x=10000)

