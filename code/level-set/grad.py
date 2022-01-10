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
nt=20
ns=100
InterpolationNum=100
timeline = UniformTimeLine(0,T,nt)

mesh = MF.boxmesh2d(domain,nx=ns,ny=ns,meshtype='tri')
space = LagrangeFiniteElementSpace(mesh,p=1)

integralalg = space.integralalg

u = space.interpolation(u)
s0 = space.interpolation(pic)

dt = timeline.dt
A1 = space.mass_matrix()

qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')

phi = space.basis(bcs)
gphi = space.grad_basis(bcs)
gdof = space.number_of_global_dofs()
ldof = space.number_of_local_dofs()
cell2dof = space.cell_to_dof()
NC = mesh.number_of_cells()
I = np.broadcast_to(cell2dof[:,:,None],shape = (NC,ldof,ldof))
J = np.broadcast_to(cell2dof[:,None,:],shape = (NC,ldof,ldof))


def grad(s): 
    ctx = DMumpsContext()
    ctx.set_silent()
    
    E0 = np.einsum('i,ijk,ijm,j -> jkm',\
            ws,gphi[...,0],phi,cellmeasure)
    E1 = np.einsum('i,ijk,ijm,j -> jkm',\
            ws,gphi[...,1],phi,cellmeasure)
    A1 = csr_matrix((E0.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    A2 = csr_matrix((E1.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    b1 = A1@s
    b2 = A2@s
    A = space.mass_matrix()
    
    ctx.set_centralized_sparse(A)
    x1 = b1.copy()
    ctx.set_rhs(x1)
    ctx.run(job=6)

    ctx.set_centralized_sparse(A)
    x2 = b2.copy()
    ctx.set_rhs(x2)
    ctx.run(job=6)

    n = space.function(dim=2)
    n[:,0] = x1[:]
    n[:,1] = x2[:]
    return n
aa = space.interpolation(pic)
n = grad(aa)
print(integralalg.L2_norm(n))
print(integralalg.L2_norm(aa.grad_value))
