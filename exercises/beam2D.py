import numpy as np

#NOTE be careful when using the Beam2D with the Truss2D because currently the
#     Truss2D is derived with only 2 DOFs per node, while the Beam2D is defined
#     with 3 DOFs per node
DOF = 3

class Beam2D(object):
    __slots__ = ['n1', 'n2', 'E', 'rho', 'Izz1', 'Izz2', 'A1', 'A2',
            'interpolation', 'le', 'thetarad']
    def __init__(self):
        self.n1 = None
        self.n2 = None
        self.E = None
        self.rho = None
        self.interpolation = 'legendre'
        self.le = None
        self.thetarad = None

def update_K_M(beam, nid_pos, ncoords, K, M, lumped=False):
    """Update K and M according to a beam element

    Properties
    ----------
    beam : Beam object
        The beam element being added to K and M
    nid_pos : dict
        Correspondence between node ids and their position in the global assembly
    ncoords : list
        Nodal coordinates of the whole model
    K : np.array
        Global stiffness matrix
    M : np.array
        Global mass matrix
    """
    pos1 = nid_pos[beam.n1]
    pos2 = nid_pos[beam.n2]
    x1, y1 = ncoords[pos1]
    x2, y2 = ncoords[pos2]
    E = beam.E
    rho = beam.rho
    Izz1 = beam.Izz1
    Izz2 = beam.Izz2
    A1 = beam.A1
    A2 = beam.A2
    le = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    beam.le = le
    beam.thetarad = np.arctan2(y2 - y1, x2 - x1)
    cosr = np.cos(beam.thetarad)
    sinr = np.sin(beam.thetarad)

    # global positions c1, c2 in the stiffness and mass matrices
    c1 = DOF*pos1
    c2 = DOF*pos2

    if beam.interpolation == 'hermitian_cubic':
        K[0+c1, 0+c1] += E*cosr**2*(A1 + A2)/(2*le) + 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c1, 1+c1] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c1, 2+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c1, 0+c2] += -E*cosr**2*(A1 + A2)/(2*le) - 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c1, 1+c2] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c1, 2+c2] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c1, 0+c1] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c1, 1+c1] += 6*E*cosr**2*(Izz1 + Izz2)/le**3 + E*sinr**2*(A1 + A2)/(2*le)
        K[1+c1, 2+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c1, 0+c2] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c1, 1+c2] += -6*E*cosr**2*(Izz1 + Izz2)/le**3 - E*sinr**2*(A1 + A2)/(2*le)
        K[1+c1, 2+c2] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c1, 0+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c1] += E*(3*Izz1 + Izz2)/le
        K[2+c1, 0+c2] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c2] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c2] += E*(Izz1 + Izz2)/le
        K[0+c2, 0+c1] += -E*cosr**2*(A1 + A2)/(2*le) - 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c2, 1+c1] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c2, 2+c1] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c2, 0+c2] += E*cosr**2*(A1 + A2)/(2*le) + 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c2, 1+c2] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c2, 2+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c2, 0+c1] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c2, 1+c1] += -6*E*cosr**2*(Izz1 + Izz2)/le**3 - E*sinr**2*(A1 + A2)/(2*le)
        K[1+c2, 2+c1] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c2, 0+c2] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c2, 1+c2] += 6*E*cosr**2*(Izz1 + Izz2)/le**3 + E*sinr**2*(A1 + A2)/(2*le)
        K[1+c2, 2+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 0+c1] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c1] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c1] += E*(Izz1 + Izz2)/le
        K[2+c2, 0+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c2] += E*(Izz1 + 3*Izz2)/le

        if not lumped:
            M[0+c1, 0+c1] += cosr**2*le*rho*(3*A1 + A2)/12 + rho*sinr**2*(10*A1*le**2 + 3*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[0+c1, 1+c1] += cosr*le*rho*sinr*(3*A1 + A2)/12 - cosr*rho*sinr*(10*A1*le**2 + 3*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[0+c1, 2+c1] += -rho*sinr*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
            M[0+c1, 0+c2] += cosr**2*le*rho*(A1 + A2)/12 + 3*rho*sinr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[0+c1, 1+c2] += cosr*le*rho*sinr*(A1 + A2)/12 - 3*cosr*rho*sinr*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[0+c1, 2+c2] += -rho*sinr*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
            M[1+c1, 0+c1] += cosr*le*rho*sinr*(3*A1 + A2)/12 - cosr*rho*sinr*(10*A1*le**2 + 3*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[1+c1, 1+c1] += cosr**2*rho*(10*A1*le**2 + 3*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le) + le*rho*sinr**2*(3*A1 + A2)/12
            M[1+c1, 2+c1] += cosr*rho*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
            M[1+c1, 0+c2] += cosr*le*rho*sinr*(A1 + A2)/12 - 3*cosr*rho*sinr*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[1+c1, 1+c2] += 3*cosr**2*rho*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le) + le*rho*sinr**2*(A1 + A2)/12
            M[1+c1, 2+c2] += cosr*rho*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
            M[2+c1, 0+c1] += -rho*sinr*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
            M[2+c1, 1+c1] += cosr*rho*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
            M[2+c1, 2+c1] += le*rho*(5*A1*le**2 + 3*A2*le**2 + 84*Izz1 + 28*Izz2)/840
            M[2+c1, 0+c2] += -rho*sinr*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
            M[2+c1, 1+c2] += cosr*rho*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
            M[2+c1, 2+c2] += -le*rho*(3*A1*le**2 + 3*A2*le**2 + 14*Izz1 + 14*Izz2)/840
            M[0+c2, 0+c1] += cosr**2*le*rho*(A1 + A2)/12 + 3*rho*sinr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[0+c2, 1+c1] += cosr*le*rho*sinr*(A1 + A2)/12 - 3*cosr*rho*sinr*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[0+c2, 2+c1] += -rho*sinr*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
            M[0+c2, 0+c2] += cosr**2*le*rho*(A1 + 3*A2)/12 + rho*sinr**2*(3*A1*le**2 + 10*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[0+c2, 1+c2] += cosr*le*rho*sinr*(A1 + 3*A2)/12 - cosr*rho*sinr*(3*A1*le**2 + 10*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[0+c2, 2+c2] += rho*sinr*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
            M[1+c2, 0+c1] += cosr*le*rho*sinr*(A1 + A2)/12 - 3*cosr*rho*sinr*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[1+c2, 1+c1] += 3*cosr**2*rho*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le) + le*rho*sinr**2*(A1 + A2)/12
            M[1+c2, 2+c1] += cosr*rho*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
            M[1+c2, 0+c2] += cosr*le*rho*sinr*(A1 + 3*A2)/12 - cosr*rho*sinr*(3*A1*le**2 + 10*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[1+c2, 1+c2] += cosr**2*rho*(3*A1*le**2 + 10*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le) + le*rho*sinr**2*(A1 + 3*A2)/12
            M[1+c2, 2+c2] += -cosr*rho*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
            M[2+c2, 0+c1] += -rho*sinr*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
            M[2+c2, 1+c1] += cosr*rho*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
            M[2+c2, 2+c1] += -le*rho*(3*A1*le**2 + 3*A2*le**2 + 14*Izz1 + 14*Izz2)/840
            M[2+c2, 0+c2] += rho*sinr*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
            M[2+c2, 1+c2] += -cosr*rho*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
            M[2+c2, 2+c2] += le*rho*(3*A1*le**2 + 5*A2*le**2 + 28*Izz1 + 84*Izz2)/840

    elif beam.interpolation == 'legendre':
        K[0+c1, 0+c1] += E*cosr**2*(A1 + A2)/(2*le) + 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c1, 1+c1] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c1, 2+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c1, 0+c2] += -E*cosr**2*(A1 + A2)/(2*le) - 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c1, 1+c2] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c1, 2+c2] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c1, 0+c1] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c1, 1+c1] += 6*E*cosr**2*(Izz1 + Izz2)/le**3 + E*sinr**2*(A1 + A2)/(2*le)
        K[1+c1, 2+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c1, 0+c2] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c1, 1+c2] += -6*E*cosr**2*(Izz1 + Izz2)/le**3 - E*sinr**2*(A1 + A2)/(2*le)
        K[1+c1, 2+c2] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c1, 0+c1] += -2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c1] += 2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c1] += E*(3*Izz1 + Izz2)/le
        K[2+c1, 0+c2] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 1+c2] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[2+c1, 2+c2] += E*(Izz1 + Izz2)/le
        K[0+c2, 0+c1] += -E*cosr**2*(A1 + A2)/(2*le) - 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c2, 1+c1] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c2, 2+c1] += 2*E*sinr*(2*Izz1 + Izz2)/le**2
        K[0+c2, 0+c2] += E*cosr**2*(A1 + A2)/(2*le) + 6*E*sinr**2*(Izz1 + Izz2)/le**3
        K[0+c2, 1+c2] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[0+c2, 2+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[1+c2, 0+c1] += -E*cosr*sinr*(A1 + A2)/(2*le) + 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c2, 1+c1] += -6*E*cosr**2*(Izz1 + Izz2)/le**3 - E*sinr**2*(A1 + A2)/(2*le)
        K[1+c2, 2+c1] += -2*E*cosr*(2*Izz1 + Izz2)/le**2
        K[1+c2, 0+c2] += E*cosr*sinr*(A1 + A2)/(2*le) - 6*E*cosr*sinr*(Izz1 + Izz2)/le**3
        K[1+c2, 1+c2] += 6*E*cosr**2*(Izz1 + Izz2)/le**3 + E*sinr**2*(A1 + A2)/(2*le)
        K[1+c2, 2+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 0+c1] += -2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c1] += 2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c1] += E*(Izz1 + Izz2)/le
        K[2+c2, 0+c2] += 2*E*sinr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 1+c2] += -2*E*cosr*(Izz1 + 2*Izz2)/le**2
        K[2+c2, 2+c2] += E*(Izz1 + 3*Izz2)/le

        if not lumped:
            M[0+c1, 0+c1] += cosr**2*le*rho*(3*A1 + A2)/12 + rho*sinr**2*(10*A1*le**2 + 3*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[0+c1, 1+c1] += cosr*le*rho*sinr*(3*A1 + A2)/12 - cosr*rho*sinr*(10*A1*le**2 + 3*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[0+c1, 2+c1] += -rho*sinr*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
            M[0+c1, 0+c2] += cosr**2*le*rho*(A1 + A2)/12 + 3*rho*sinr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[0+c1, 1+c2] += cosr*le*rho*sinr*(A1 + A2)/12 - 3*cosr*rho*sinr*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[0+c1, 2+c2] += -rho*sinr*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
            M[1+c1, 0+c1] += cosr*le*rho*sinr*(3*A1 + A2)/12 - cosr*rho*sinr*(10*A1*le**2 + 3*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[1+c1, 1+c1] += cosr**2*rho*(10*A1*le**2 + 3*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le) + le*rho*sinr**2*(3*A1 + A2)/12
            M[1+c1, 2+c1] += cosr*rho*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
            M[1+c1, 0+c2] += cosr*le*rho*sinr*(A1 + A2)/12 - 3*cosr*rho*sinr*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[1+c1, 1+c2] += 3*cosr**2*rho*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le) + le*rho*sinr**2*(A1 + A2)/12
            M[1+c1, 2+c2] += cosr*rho*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
            M[2+c1, 0+c1] += -rho*sinr*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
            M[2+c1, 1+c1] += cosr*rho*(15*A1*le**2 + 7*A2*le**2 + 42*Izz2)/420
            M[2+c1, 2+c1] += le*rho*(5*A1*le**2 + 3*A2*le**2 + 84*Izz1 + 28*Izz2)/840
            M[2+c1, 0+c2] += -rho*sinr*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
            M[2+c1, 1+c2] += cosr*rho*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
            M[2+c1, 2+c2] += -le*rho*(3*A1*le**2 + 3*A2*le**2 + 14*Izz1 + 14*Izz2)/840
            M[0+c2, 0+c1] += cosr**2*le*rho*(A1 + A2)/12 + 3*rho*sinr**2*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[0+c2, 1+c1] += cosr*le*rho*sinr*(A1 + A2)/12 - 3*cosr*rho*sinr*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[0+c2, 2+c1] += -rho*sinr*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
            M[0+c2, 0+c2] += cosr**2*le*rho*(A1 + 3*A2)/12 + rho*sinr**2*(3*A1*le**2 + 10*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[0+c2, 1+c2] += cosr*le*rho*sinr*(A1 + 3*A2)/12 - cosr*rho*sinr*(3*A1*le**2 + 10*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[0+c2, 2+c2] += rho*sinr*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
            M[1+c2, 0+c1] += cosr*le*rho*sinr*(A1 + A2)/12 - 3*cosr*rho*sinr*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le)
            M[1+c2, 1+c1] += 3*cosr**2*rho*(3*A1*le**2 + 3*A2*le**2 - 28*Izz1 - 28*Izz2)/(140*le) + le*rho*sinr**2*(A1 + A2)/12
            M[1+c2, 2+c1] += cosr*rho*(6*A1*le**2 + 7*A2*le**2 - 42*Izz2)/420
            M[1+c2, 0+c2] += cosr*le*rho*sinr*(A1 + 3*A2)/12 - cosr*rho*sinr*(3*A1*le**2 + 10*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le)
            M[1+c2, 1+c2] += cosr**2*rho*(3*A1*le**2 + 10*A2*le**2 + 21*Izz1 + 21*Izz2)/(35*le) + le*rho*sinr**2*(A1 + 3*A2)/12
            M[1+c2, 2+c2] += -cosr*rho*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
            M[2+c2, 0+c1] += -rho*sinr*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
            M[2+c2, 1+c1] += cosr*rho*(-7*A1*le**2 - 6*A2*le**2 + 42*Izz1)/420
            M[2+c2, 2+c1] += -le*rho*(3*A1*le**2 + 3*A2*le**2 + 14*Izz1 + 14*Izz2)/840
            M[2+c2, 0+c2] += rho*sinr*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
            M[2+c2, 1+c2] += -cosr*rho*(7*A1*le**2 + 15*A2*le**2 + 42*Izz1)/420
            M[2+c2, 2+c2] += le*rho*(3*A1*le**2 + 5*A2*le**2 + 28*Izz1 + 84*Izz2)/840

    else:
        raise NotImplementedError('beam interpolation "%s" not implemented' % beam.interpolation)

    if lumped:
        M[0+c1, 0+c1] += le*rho*(3*A1 + A2)*(cosr**2 + sinr**2)/8
        M[1+c1, 1+c1] += le*rho*(3*A1 + A2)*(cosr**2 + sinr**2)/8
        M[2+c1, 2+c1] += le*(5*A1*le**2*rho + 3*A2*le**2*rho + 72*Izz1 + 24*Izz2)/192
        M[0+c2, 0+c2] += le*rho*(A1 + 3*A2)*(cosr**2 + sinr**2)/8
        M[1+c2, 1+c2] += le*rho*(A1 + 3*A2)*(cosr**2 + sinr**2)/8
        M[2+c2, 2+c2] += le*(3*A1*le**2*rho + 5*A2*le**2*rho + 24*Izz1 + 72*Izz2)/192
