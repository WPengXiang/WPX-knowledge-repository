#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: level.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年11月19日 星期五 11时42分52秒
'''  
import numpy as np
from mumps import DMumpsContext
import scipy
from scipy.linalg import solve

from scipy import interpolate
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric
from fealpy.boundarycondition import DirichletBC
from fealpy.quadrature import GaussLegendreQuadrature
from scipy.sparse import bmat,csr_matrix,hstack,vstack,spdiags
import matplotlib
from fealpy.tools.show import showmultirate

domain = [0,1,0,1]

@cartesian
def u(p):
    x = p[...,0]
    y = p[...,1]
    u = np.zeros(p.shape)
    u[...,0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[...,1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u


@cartesian
def pic(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    #val = 1/(1 + np.exp(val/1))
    return val



T=2
nt=100
ns=100
InterpolationNum=100
timeline = UniformTimeLine(0,T,nt)

mesh = MF.boxmesh2d(domain,nx=ns,ny=ns,meshtype='tri')
space = LagrangeFiniteElementSpace(mesh,p=2)

u = space.interpolation(u)
s0 = space.interpolation(pic)
s1 = space.function()

'''
fig1 = plt.figure()
node = mesh.node
x = tuple(node[:,0])
y = tuple(node[:,1])
NN = mesh.number_of_nodes()
u = u[:NN]
ux = tuple(u[:,0])
uy = tuple(u[:,1])

o = ux
norm = matplotlib.colors.Normalize()
cm = matplotlib.cm.copper
sm = matplotlib.cm.ScalarMappable(cmap=cm,norm=norm)
sm.set_array([])
plt.quiver(x,y,ux,uy,color=cm(norm(o)))
plt.colorbar(sm)
plt.show()
'''
dt = timeline.dt
A1 = space.mass_matrix()

qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')

phi = space.basis(bcs)
gphi = space.grad_basis(bcs)
gdof = space.number_of_global_dofs()
cell2dof = space.cell_to_dof()

E0 = np.einsum('i,ijk,ijm,ij,j -> jkm',ws,phi,gphi[...,0],u(bcs)[...,0],cellmeasure)
E1 = np.einsum('i,ijk,ijm,ij,j -> jkm',ws,phi,gphi[...,1],u(bcs)[...,1],cellmeasure)
E = E0+E1
I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
A2 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))

A = A1-dt/2*A2

ctx = DMumpsContext()
ctx.set_silent()
fname = './test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['uh'] = s0
mesh.to_vtk(fname=fname)

for i in range(nt):
    
    t1 = timeline.next_time_level()
    #print("t1=",t1)
    b = A@s0+dt/2*A2@s0
    
    ctx.set_centralized_sparse(A)
    x = b.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)

    s0[:] = x[:]
    fname = './test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['uh'] = s0
    mesh.to_vtk(fname=fname)
ctx.destroy()

