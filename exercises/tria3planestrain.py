import numpy as np

DOF = 2

class Tria3PlaneStrainIso(object):
    __slots__ = ['n1', 'n2', 'n3', 'E', 'nu', 'A', 'h', 'rho']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        self.n3 = None
        # Material Lastrobe Lescalloy
        self.E = None
        self.nu = None
        self.rho = None

def update_K_M(tria, nid_pos, ncoords, K, M):
    """Update a global stiffness matrix K and mass matrix M

    Properties
    ----------
    tria : `.Tria3PlaneStrainIso` object
        The Tria3PlaneStrainIso element being added to K
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    K : np.array
        Global stiffness matrix updated in-place
    M : np.array
        Global mass matrix updated in-place
    """
    pos1 = nid_pos[tria.n1]
    pos2 = nid_pos[tria.n2]
    pos3 = nid_pos[tria.n3]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    x3, y3 = ncoords[pos3]
    A = abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))/2)
    tria.A = A
    E = tria.E
    nu = tria.nu
    h = tria.h
    rho = tria.rho

    # positions the global matrices
    c1 = DOF*pos1
    c2 = DOF*pos2
    c3 = DOF*pos3

    K[0+c1, 0+c1] += 0.25*E*h*((nu - 1)*(y2 - y3)**2 + (2*nu - 1)*(x2 - x3)**2)/(A*(nu + 1)*(2*nu - 1))
    K[0+c1, 1+c1] += 0.25*E*h*(1 - nu)*(x2 - x3)*(y2 - y3)/(A*(nu + 1)*(2*nu - 1))
    K[0+c1, 0+c2] += -0.25*E*h*((nu - 1)*(y1 - y3)*(y2 - y3) + (2*nu - 1)*(x1 - x3)*(x2 - x3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c1, 1+c2] += -0.25*E*h*(nu*(x1 - x3)*(y2 - y3) - (2*nu - 1)*(x2 - x3)*(y1 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c1, 0+c3] += 0.25*E*h*((nu - 1)*(y1 - y2)*(y2 - y3) + (2*nu - 1)*(x1 - x2)*(x2 - x3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c1, 1+c3] += 0.25*E*h*(nu*(x1 - x2)*(y2 - y3) - (2*nu - 1)*(x2 - x3)*(y1 - y2))/(A*(nu + 1)*(2*nu - 1))
    K[1+c1, 0+c1] += 0.25*E*h*(1 - nu)*(x2 - x3)*(y2 - y3)/(A*(nu + 1)*(2*nu - 1))
    K[1+c1, 1+c1] += 0.25*E*h*((nu - 1)*(x2 - x3)**2 + (2*nu - 1)*(y2 - y3)**2)/(A*(nu + 1)*(2*nu - 1))
    K[1+c1, 0+c2] += -0.25*E*h*(nu*(x2 - x3)*(y1 - y3) - (2*nu - 1)*(x1 - x3)*(y2 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[1+c1, 1+c2] += -0.25*E*h*((nu - 1)*(x1 - x3)*(x2 - x3) + (2*nu - 1)*(y1 - y3)*(y2 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[1+c1, 0+c3] += 0.25*E*h*(nu*(x2 - x3)*(y1 - y2) - (2*nu - 1)*(x1 - x2)*(y2 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[1+c1, 1+c3] += 0.25*E*h*((nu - 1)*(x1 - x2)*(x2 - x3) + (2*nu - 1)*(y1 - y2)*(y2 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c2, 0+c1] += -0.25*E*h*((nu - 1)*(y1 - y3)*(y2 - y3) + (2*nu - 1)*(x1 - x3)*(x2 - x3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c2, 1+c1] += -0.25*E*h*(nu*(x2 - x3)*(y1 - y3) - (2*nu - 1)*(x1 - x3)*(y2 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c2, 0+c2] += 0.25*E*h*((nu - 1)*(y1 - y3)**2 + (2*nu - 1)*(x1 - x3)**2)/(A*(nu + 1)*(2*nu - 1))
    K[0+c2, 1+c2] += 0.25*E*h*(1 - nu)*(x1 - x3)*(y1 - y3)/(A*(nu + 1)*(2*nu - 1))
    K[0+c2, 0+c3] += -0.25*E*h*((nu - 1)*(y1 - y2)*(y1 - y3) + (2*nu - 1)*(x1 - x2)*(x1 - x3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c2, 1+c3] += -0.25*E*h*(nu*(x1 - x2)*(y1 - y3) - (2*nu - 1)*(x1 - x3)*(y1 - y2))/(A*(nu + 1)*(2*nu - 1))
    K[1+c2, 0+c1] += -0.25*E*h*(nu*(x1 - x3)*(y2 - y3) - (2*nu - 1)*(x2 - x3)*(y1 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[1+c2, 1+c1] += -0.25*E*h*((nu - 1)*(x1 - x3)*(x2 - x3) + (2*nu - 1)*(y1 - y3)*(y2 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[1+c2, 0+c2] += 0.25*E*h*(1 - nu)*(x1 - x3)*(y1 - y3)/(A*(nu + 1)*(2*nu - 1))
    K[1+c2, 1+c2] += 0.25*E*h*((nu - 1)*(x1 - x3)**2 + (2*nu - 1)*(y1 - y3)**2)/(A*(nu + 1)*(2*nu - 1))
    K[1+c2, 0+c3] += -0.25*E*h*(nu*(x1 - x3)*(y1 - y2) - (2*nu - 1)*(x1 - x2)*(y1 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[1+c2, 1+c3] += -0.25*E*h*((nu - 1)*(x1 - x2)*(x1 - x3) + (2*nu - 1)*(y1 - y2)*(y1 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c3, 0+c1] += 0.25*E*h*((nu - 1)*(y1 - y2)*(y2 - y3) + (2*nu - 1)*(x1 - x2)*(x2 - x3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c3, 1+c1] += 0.25*E*h*(nu*(x2 - x3)*(y1 - y2) - (2*nu - 1)*(x1 - x2)*(y2 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c3, 0+c2] += -0.25*E*h*((nu - 1)*(y1 - y2)*(y1 - y3) + (2*nu - 1)*(x1 - x2)*(x1 - x3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c3, 1+c2] += -0.25*E*h*(nu*(x1 - x3)*(y1 - y2) - (2*nu - 1)*(x1 - x2)*(y1 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[0+c3, 0+c3] += 0.25*E*h*((nu - 1)*(y1 - y2)**2 + (2*nu - 1)*(x1 - x2)**2)/(A*(nu + 1)*(2*nu - 1))
    K[0+c3, 1+c3] += 0.25*E*h*(1 - nu)*(x1 - x2)*(y1 - y2)/(A*(nu + 1)*(2*nu - 1))
    K[1+c3, 0+c1] += 0.25*E*h*(nu*(x1 - x2)*(y2 - y3) - (2*nu - 1)*(x2 - x3)*(y1 - y2))/(A*(nu + 1)*(2*nu - 1))
    K[1+c3, 1+c1] += 0.25*E*h*((nu - 1)*(x1 - x2)*(x2 - x3) + (2*nu - 1)*(y1 - y2)*(y2 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[1+c3, 0+c2] += -0.25*E*h*(nu*(x1 - x2)*(y1 - y3) - (2*nu - 1)*(x1 - x3)*(y1 - y2))/(A*(nu + 1)*(2*nu - 1))
    K[1+c3, 1+c2] += -0.25*E*h*((nu - 1)*(x1 - x2)*(x1 - x3) + (2*nu - 1)*(y1 - y2)*(y1 - y3))/(A*(nu + 1)*(2*nu - 1))
    K[1+c3, 0+c3] += 0.25*E*h*(1 - nu)*(x1 - x2)*(y1 - y2)/(A*(nu + 1)*(2*nu - 1))
    K[1+c3, 1+c3] += 0.25*E*h*((nu - 1)*(x1 - x2)**2 + (2*nu - 1)*(y1 - y2)**2)/(A*(nu + 1)*(2*nu - 1))

    M[0+c1, 0+c1] += 0.166666666666667*A*h*rho
    M[0+c1, 1+c1] += 0
    M[0+c1, 0+c2] += 0.0833333333333333*A*h*rho
    M[0+c1, 1+c2] += 0
    M[0+c1, 0+c3] += 0.0833333333333333*A*h*rho
    M[0+c1, 1+c3] += 0
    M[1+c1, 0+c1] += 0
    M[1+c1, 1+c1] += 0.166666666666667*A*h*rho
    M[1+c1, 0+c2] += 0
    M[1+c1, 1+c2] += 0.0833333333333333*A*h*rho
    M[1+c1, 0+c3] += 0
    M[1+c1, 1+c3] += 0.0833333333333333*A*h*rho
    M[0+c2, 0+c1] += 0.0833333333333333*A*h*rho
    M[0+c2, 1+c1] += 0
    M[0+c2, 0+c2] += 0.166666666666667*A*h*rho
    M[0+c2, 1+c2] += 0
    M[0+c2, 0+c3] += 0.0833333333333333*A*h*rho
    M[0+c2, 1+c3] += 0
    M[1+c2, 0+c1] += 0
    M[1+c2, 1+c1] += 0.0833333333333333*A*h*rho
    M[1+c2, 0+c2] += 0
    M[1+c2, 1+c2] += 0.166666666666667*A*h*rho
    M[1+c2, 0+c3] += 0
    M[1+c2, 1+c3] += 0.0833333333333333*A*h*rho
    M[0+c3, 0+c1] += 0.0833333333333333*A*h*rho
    M[0+c3, 1+c1] += 0
    M[0+c3, 0+c2] += 0.0833333333333333*A*h*rho
    M[0+c3, 1+c2] += 0
    M[0+c3, 0+c3] += 0.166666666666667*A*h*rho
    M[0+c3, 1+c3] += 0
    M[1+c3, 0+c1] += 0
    M[1+c3, 1+c1] += 0.0833333333333333*A*h*rho
    M[1+c3, 0+c2] += 0
    M[1+c3, 1+c2] += 0.0833333333333333*A*h*rho
    M[1+c3, 0+c3] += 0
    M[1+c3, 1+c3] += 0.166666666666667*A*h*rho

