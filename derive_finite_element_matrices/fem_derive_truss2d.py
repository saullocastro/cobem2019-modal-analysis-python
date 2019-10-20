import numpy as np
import sympy
from sympy import Matrix

DOF = 2

sympy.var('A, le, xi')
sympy.var('rho, E, nu, c, s')

N1 = (1-xi)/2
N2 = (1+xi)/2

Nu = Matrix([[N1, 0, N2, 0]])
Nv = Matrix([[0, N1, 0, N2]])

# Constitutive linear stiffness matrix
T = Matrix([[c, s, 0, 0],
            [-s, c, 0, 0],
            [0, 0, c, s],
            [0, 0, -s, c]])

num_nodes = 2

Ke = A*E/le*Matrix([[1, 0, -1, 0],
                   [0, 0, 0, 0],
                   [-1, 0, 1, 0],
                   [0, 0, 0, 0]])

Me = (le/2)*sympy.integrate(rho*A*(Nu.T*Nu + Nv.T*Nv), (xi, -1, +1))
Me_lumped = A*rho*le/2*Matrix(
                  [[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

K = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
M = sympy.zeros(num_nodes*DOF, num_nodes*DOF)
M_lumped = sympy.zeros(num_nodes*DOF, num_nodes*DOF)

# global
K[:, :] = T.T*Ke*T
M[:, :] = T.T*Me*T
M_lumped[:, :] = T.T*Me_lumped*T

# K represents the global stiffness matrix
# in case we want to apply coordinate transformations

def name_ind(i):
    if i >=0 and i < DOF:
        return 'c1'
    elif i >= DOF and i < 2*DOF:
        return 'c2'
    elif i >= 2*DOF and i < 3*DOF:
        return 'c3'
    else:
        raise

print('printing K')
for ind, val in np.ndenumerate(K):
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    K[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', K[ind])
print('printing M')
for ind, val in np.ndenumerate(M):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M[ind])
print('printing M_lumped')
for ind, val in np.ndenumerate(M_lumped):
    if val == 0:
        continue
    i, j = ind
    si = name_ind(i)
    sj = name_ind(j)
    print('    M[%d+%s, %d+%s]' % (i%DOF, si, j%DOF, sj), '+=', M_lumped[ind])
