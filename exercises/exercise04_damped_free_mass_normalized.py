import sympy
from sympy import Function, dsolve, Symbol

# symbols
t = Symbol('t', positive=True)
wn = Symbol('\omega_n', positive=True)
zeta = Symbol('\zeta')

# unknown function
u = Function('u')(t)

# solving ODE
sol = dsolve(u.diff(t, t) + 2*zeta*wn*u.diff(t) + wn**2*u)

# sol.lhs ==> u(t)
# sol.rhs ==> solution
print(sol.rhs.simplify())
print()
sympy.print_latex(sol.rhs.simplify())

