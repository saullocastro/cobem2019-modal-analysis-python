import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# number of nodes in each direction
nx = 20
ny = 4

# geometry
a = 10
b = 1

# creating mesh
xtmp = np.linspace(0, a, nx)
ytmp = np.linspace(0, b, ny)
xmesh, ymesh = np.meshgrid(xtmp, ytmp)
ncoords = np.vstack((xmesh.T.flatten(), ymesh.T.flatten())).T

# triangulation to establish nodal connectivity
d = Delaunay(ncoords)

plt.clf()
ax = plt.gca()
ax.set_aspect('equal')
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
plt.triplot(ncoords[:, 0], ncoords[:, 1], d.simplices, lw=0.5)
plt.plot(ncoords[:, 0], ncoords[:, 1], 'o', ms=2)
plt.show()

