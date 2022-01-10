import numpy as np

from fealpy.decorator import cartesian

class Poisuille:
    """
    [0, 1]^3
    u(x, y) = (4y(1-y), 0)
    p = 8(1-x)
    """
    def __init__(self,eps=1e-12):
        self.eps = eps
        self.box = [0, 1, 0, 1, 0, 1]

    def domain(self):
        return self.box

    @cartesian
    def velocity(self, p):
        x = p[...,0]
        y = p[...,1]
        z = p[...,2]
        value = np.zeros(p.shape,dtype=np.float)
        value[...,0] = 4*y*(1-y)
        return value

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        val = 8*(1-x) 
        return val
    
    @cartesian
    def source(self, p):
        val = np.zeros(p.shape, dtype=np.float)
        return val

    @cartesian
    def is_p_boundary(self,p):
        return (np.abs(p[..., 0]) < self.eps) | (np.abs(p[..., 0] - 1.0) < self.eps)
      
    @cartesian
    def is_wall_boundary(self,p):
        return (np.abs(p[..., 1]) < self.eps) | (np.abs(p[..., 1] - 1.0) < self.eps)

    @cartesian
    def p_dirichlet(self, p):
        return self.pressure(p)
    
    @cartesian
    def u_dirichlet(self, p):
        return self.velocity(p)
