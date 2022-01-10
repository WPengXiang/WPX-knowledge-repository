import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian
@cartesian
def u(p):
    x = p[...,0]
    y = p[...,1]
    return x*x+y*y

p = 1 
n = 2

domain = np.array([0,1,0,1])
mesh = MF.boxmesh2d(domain, nx=n, ny=n, meshtype='tri')

errorMatrix = np.zeros(4, dtype=mesh.ftype)
for i in range(4):
    space = LagrangeFiniteElementSpace(mesh, p=p)
    uh = space.function() 
    #A = space.stiff_matrix()
    M = space.mass_matrix()

    uh[:] = space.projection(u)

    errorMatrix[i] = space.integralalg.L2_error(u, uh.value)

    if i < 3:
        mesh.uniform_refine()


fig = plt.figure()
axes = fig.gca(projection='3d')
uh.add_plot(axes)
print(errorMatrix)

fig = plt.figure()
plt.plot([0,1,2,3],errorMatrix)
plt.xticks([0,1,2,3])
plt.show()
