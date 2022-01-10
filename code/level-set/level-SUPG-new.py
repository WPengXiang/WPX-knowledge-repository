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
    val = 1/(1 + np.exp(val/1))
    return val
def pic1(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    val = np.exp(val)/(1 + np.exp(val/1))**2
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
n = space.function(dim=2)

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

norm_u = integralalg.L2_norm(u)
h = max(mesh.entity_measure('cell'))

B0 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
        ws,gphi[...,0],u(bcs)[...,0],gphi[...,0],u(bcs)[...,0],cellmeasure)
B1 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
        ws,gphi[...,1],u(bcs)[...,1],gphi[...,1],u(bcs)[...,1],cellmeasure)
B = E0+E1
I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
A3 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))

A = A1 - dt/2*A2 + (h/(2*norm_u)) *(A2 + (dt/2)*A3) 

ctx = DMumpsContext()
ctx.set_silent()

fname = './test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['uh'] = s0
mesh.to_vtk(fname=fname)

for i in range(nt):
    
    t1 = timeline.next_time_level()
    #print("t1=",t1)
    b = A@s0 + dt/2*A2@s0 + (h/(2*norm_u))*(A2@s0 - (dt/2)*A3@s0) 

    ctx.set_centralized_sparse(A)
    x = b.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)

    s0[:] = x[:]
    fname = './test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['uh'] = s0
    mesh.to_vtk(fname=fname)
ctx.destroy()

def re(s):
    ctx = DMumpsContext()
    ctx.set_silent()
    
    norm = integralalg.L2_norm(s)
    print(norm)
    n = s.grad_value

    epsilon=1
    dt=0.001
    A0 = space.mass_matrix()

    A11 = np.einsum('i,ijk,ijm,ij,j -> jkm',\
            ws,phi,gphi[...,0],n(bcs)[...,0],cellmeasure)
    A12 = np.einsum('i,ijk,ijm,ij,j -> jkm',\
            ws,phi,gphi[...,1],n(bcs)[...,1],cellmeasure)
    A1 = A11+A12
    A1 = csr_matrix((A1.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    
    A21 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
            ws,gphi[...,0],n(bcs)[...,0],gphi[...,0],n(bcs)[...,0],cellmeasure)
    A22 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
            ws,gphi[...,1],n(bcs)[...,1],gphi[...,1],n(bcs)[...,1],cellmeasure)
    A23 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
            ws,gphi[...,1],n(bcs)[...,1],gphi[...,0],n(bcs)[...,0],cellmeasure)
    A24 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
            ws,gphi[...,0],n(bcs)[...,0],gphi[...,1],n(bcs)[...,1],cellmeasure)
    A2 = A21+A22+A23+A24
    A2 = csr_matrix((A2.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    
    A31 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
            ws,phi,s(bcs),gphi[...,0],n(bcs)[...,0],cellmeasure)
    A32 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
            ws,phi,s(bcs),gphi[...,1],n(bcs)[...,1],cellmeasure)
    A3 = A31+A32
    A3 = csr_matrix((A3.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    
    A = norm*(A0) - (dt/2)*A1 + (epsilon*dt/2)*A2 + epsilon*dt*A3
    
    while True:
        b = norm*(A0@s)+(dt/2)*A1*s -(epsilon*dt/2)*A2@s

        ctx.set_centralized_sparse(A)
        x = b.copy()
        ctx.set_rhs(x)
        ctx.run(job=6)
        
        error = space.function()
        error[:]=np.abs(x[:]-s[:])
        error = integralalg.L2_norm(error)
        s[:] = x[:]
        print(error)
        if error <= 0.000001:
            ctx.destroy()
            print(integralalg.L2_norm(s.grad_value))
            return s
            break

aa = space.interpolation(pic)
print(integralalg.L2_norm(aa.grad_value))
re(s0)
