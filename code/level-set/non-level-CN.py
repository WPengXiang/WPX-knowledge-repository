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
nt=50
ns=100
InterpolationNum=100
timeline = UniformTimeLine(0,T,nt)

mesh = MF.boxmesh2d(domain,nx=ns,ny=ns,meshtype='tri')
space = LagrangeFiniteElementSpace(mesh,p=1)

u = space.interpolation(u)
s0 = space.interpolation(pic)

dt = timeline.dt
A1 = space.mass_matrix()

qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')
integralalg = space.integralalg

phi = space.basis(bcs)
gphi = space.grad_basis(bcs)
gdof = space.number_of_global_dofs()
cell2dof = space.cell_to_dof()

E0 = np.einsum('i,ijk,ij,ijm,j -> jkm',ws,phi,u(bcs)[...,0],gphi[...,0],cellmeasure)
E1 = np.einsum('i,ijk,ij,ijm,j -> jkm',ws,phi,u(bcs)[...,1],gphi[...,1],cellmeasure)
E = E0+E1
I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
A2 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))

A = A1+dt/2*A2


for i in range(nt):
    ctx = DMumpsContext()
    ctx.set_silent()
    fname = './test_'+ str(0).zfill(10) + '.vtu'
    mesh.nodedata['uh'] = s0
    mesh.to_vtk(fname=fname)
        
    t1 = timeline.next_time_level()
    #print("t1=",t1)
    b = A@s0 - dt/2*A2@s0
    
    ctx.set_centralized_sparse(A)
    x = b.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)

    s0[:] = x[:]
    fname = './test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['uh'] = s0
    mesh.to_vtk(fname=fname)
ctx.destroy()

def re(s0):
    print(integralalg.L2_norm(s0.grad_value))
    ctx = DMumpsContext()
    ctx.set_silent()
    alpha = 6.25
    s1 = s0
    dt = 0.001
    A = 1/dt*space.mass_matrix()
    while True:
        bb = np.einsum('i,ijk,ij,ij,j->jk',ws,phi,\
                  1 - np.sqrt((s1.grad_value(bcs)[...,0])**2+(s1.grad_value(bcs)[...,1])**2),\
                      np.sign(s0(bcs)),cellmeasure)
        b = np.zeros((gdof))
        np.add.at(b,cell2dof,bb)
        b = b+A@s1 - alpha*space.stiff_matrix()@s1
        
        
        ctx.set_centralized_sparse(A)
        x = b.copy()
        ctx.set_rhs(x)
        ctx.run(job=6)
 
        error = space.function()
        error[:]=np.abs(x[:]-s1[:])
        error = integralalg.L2_norm(error)
        s1[:] = x[:]
        print(error)
        print("asdadsadad",integralalg.L2_norm(s1.grad_value))
        if error <= 0.0001:
            print(integralalg.L2_norm(s1.grad_value))
            ctx.destroy()
            return s1
            break

re(s0)





