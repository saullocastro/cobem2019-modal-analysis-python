import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
m = Symbol('m', positive=True)
k = Symbol('k', positive=True)
F0 = Symbol('F0', positive=True)
wf = Symbol('\omega_f', positive=True)

# unknown function
u = Function('u')(t)

# solving ODE
F = F0*sympy.sin(wf*t)
sol = dsolve(m*u.diff(t, t) + k*u - F)

# sol.lhs ==> u(t)
# sol.rhs ==> solution
print(sol.rhs)

wn = sympy.Symbol('\omega_n')
expr = sol.rhs.subs(sympy.sqrt(k/m), wn)
print(expr)


