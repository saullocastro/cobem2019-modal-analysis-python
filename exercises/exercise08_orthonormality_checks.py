import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import Delaunay
from scipy.linalg import eigh

from truss2d import Truss2D, update_K_M

DOF = 2

lumped = True

# number of nodes in each direction
nx = 20
ny = 4

# geometry
a = 10
b = 1
A = 0.01**2

# material properties
E = 70e9
rho = 2.6e3

# creating mesh
xtmp = np.linspace(0, a, nx)
ytmp = np.linspace(0, b, ny)
xmesh, ymesh = np.meshgrid(xtmp, ytmp)
ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T
x = ncoords[:, 0]
y = ncoords[:, 1]
nid_pos = dict(zip(np.arange(len(ncoords)), np.arange(len(ncoords))))

# triangulation to establish nodal connectivity
d = Delaunay(ncoords)

# extracting edges out of triangulation to form the truss elements
edges = {}
for s in d.simplices:
    edges[tuple(sorted([s[0], s[1]]))] = [s[0], s[1]]
    edges[tuple(sorted([s[1], s[2]]))] = [s[1], s[2]]
    edges[tuple(sorted([s[2], s[0]]))] = [s[2], s[0]]
nAnBs = np.array([list(edge) for edge in edges.values()], dtype=int)

#NOTE using dense matrices
K = np.zeros((DOF*nx*ny, DOF*nx*ny))
M = np.zeros((DOF*nx*ny, DOF*nx*ny))

# creating truss elements
elems = []
for n1, n2 in nAnBs:
    elem = Truss2D()
    elem.n1 = n1
    elem.n2 = n2
    elem.E = E
    elem.A = A
    elem.rho = rho
    update_K_M(elem, nid_pos, ncoords, K, M, lumped=lumped)
    elems.append(elem)
if lumped:
    assert np.count_nonzero(M-np.diag(np.diagonal(M))) == 0

# applying boundary conditions
bk = np.zeros(K.shape[0], dtype=bool) # defining known DOFs
check = np.isclose(x, 0.)
bk[0::DOF] = check
bk[1::DOF] = check
bu = ~bk # defining unknown DOFs

# sub-matrices corresponding to unknown DOFs
Kuu = K[bu, :][:, bu]
Muu = M[bu, :][:, bu]

# solving
# NOTE: extracting ALL eigenvectors
eigvals, U = eigh(a=Kuu, b=Muu)
wn = eigvals**0.5

print('eigenvalues (wn**2)', wn[:3]**2)
for I, J in [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]:
    print('I =', I, 'J =', J,
          '\tUI . UJ %1.2f' % (U[:, I] @ U[:, J]),
          '\tUI M UJ %1.2f' % (U[:, I] @ Muu @ U[:, J]),
          '\tUI K UJ %1.2f' % (U[:, I] @ Kuu @ U[:, J]))

