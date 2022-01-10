import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.mesh import HalfEdgeMesh2d
def u(p):
    x = p[...,0]
    y = p[...,1]
    return x*x+y*y

p = 1 
n = 3

domain = np.array([0,1,0,1])
mesh = MF.boxmesh2d(domain, nx=n, ny=n, meshtype='poly')
mesh = HalfEdgeMesh2d.from_mesh(mesh)
mesh.init_level_info()
errorMatrix = np.zeros(4, dtype=mesh.ftype)
for i in range(4):
    space = ScaledMonomialSpace2d(mesh, p=p)
    uh = space.function() 
    A = space.stiff_matrix()
    M = space.mass_matrix()
    uh[:] = space.local_projection(u)

    errorMatrix[i] = space.integralalg.L2_error(u, uh.value)

    if i < 3:
        mesh.uniform_refine()

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)

fig = plt.figure()
plt.plot([0,1,2,3],errorMatrix)
plt.xticks([0,1,2,3])
plt.show()
