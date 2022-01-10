
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric
from fealpy.mesh import LagrangeTriangleMesh

@cartesian
def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    return val

domain = [0, 1, 0, 1]
mesh = MF.boxmesh2d(domain, nx=2, ny=2, meshtype='tri')

space = LagrangeFiniteElementSpace(mesh, p=3)
ip = space.interpolation_points()

node = mesh.entity('node')
cell = mesh.entity('cell')
lmesh = LagrangeTriangleMesh(node, cell, p=3)



fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ip, showindex=True)

fig = plt.figure()
axes = fig.gca()
node = lmesh.entity('node')
mesh.add_plot(axes)
mesh.find_node(axes, node=node, showindex=True)
plt.show()


