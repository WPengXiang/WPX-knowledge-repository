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
from scipy.sparse.linalg import spsolve

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
    val = 1/(1 + np.exp(-val/epsilon))
    return val

T=np.pi/4
nt=400
ns=80
dx = 1/ns
timeline = UniformTimeLine(0,T,nt)
epsilon = (dx**0.9)/2 
mesh = MF.boxmesh2d(domain,nx=ns,ny=ns,meshtype='tri')
space = LagrangeFiniteElementSpace(mesh,p=1)

'''
fig1 = plt.figure()
node = mesh.node
x = tuple(node[:,0])
y = tuple(node[:,1])
NN = mesh.number_of_nodes()
u = u(node)[:NN]
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


integralalg = space.integralalg

u = space.interpolation(u)
s0 = space.interpolation(pic)

dt = timeline.dt

qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')

def SUPG(u,s0,space): 
    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    s1 = space.function() 
    
    norm_u = np.sum(integralalg.L2_norm(u,celltype=True),axis=1)
    h = mesh.entity_measure('cell')
    s = h/(2*norm_u) 
    
    A1 = space.mass_matrix()
    E0 = np.einsum('i,ijk,ij,ijm,j -> jkm',ws,gphi[...,0],u(bcs)[...,0],phi,cellmeasure)
    E1 = np.einsum('i,ijk,ij,ijm,j -> jkm',ws,gphi[...,1],u(bcs)[...,1],phi,cellmeasure)
    E = E0+E1
    I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
    A2 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    
    E0 = np.einsum('i,ijk,ij,ijm,j,j -> jkm',\
            ws,gphi[...,0],u(bcs)[...,0],phi,cellmeasure,s)
    E1 = np.einsum('i,ijk,ij,ijm,j,j -> jkm',\
            ws,gphi[...,1],u(bcs)[...,1],phi,cellmeasure,s)
    E = E0+E1
    I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
    A3 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    
    E0 = np.einsum('i,ij,ijk,ijm,ij,j,j -> jkm',ws,u(bcs)[...,0],gphi[...,0]\
            ,gphi[...,0],u(bcs)[...,0],cellmeasure,s)
    E1 = np.einsum('i,ij,ijk,ijm,ij,j,j -> jkm',ws,u(bcs)[...,1],gphi[...,1]\
            ,gphi[...,1],u(bcs)[...,1],cellmeasure,s)
    E2 = np.einsum('i,ij,ijk,ijm,ij,j,j -> jkm',ws,u(bcs)[...,0],gphi[...,0]\
            ,gphi[...,1],u(bcs)[...,1],cellmeasure,s)
    E3 = np.einsum('i,ij,ijk,ijm,ij,j,j -> jkm',ws,u(bcs)[...,1],gphi[...,1]\
            ,gphi[...,0],u(bcs)[...,0],cellmeasure,s)
    E = E0+E1+E2+E3
    I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
    A4 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))

    A = A1 - dt/2*A2 + A3 + dt/2*A4 

    b = A1@s0 + dt/2*A2@s0 + A3@s0 - dt/2*A4@s0 
    
    x = spsolve(A,b)
    s1[:] = x[:]
    return s1

def grads(s,space):
    phi = space.basis(bcs)
    gphi = s.grad_value(bcs)
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()

    A = space.mass_matrix()
    
    s11 = s.grad_value(bcs)[...,0]
    b11 = np.einsum('i,ijk,ij,j -> jk',ws,phi,s11,cellmeasure)
    b1 = np.zeros(gdof)
    np.add.at(b1,cell2dof,b11)
    s12 = s.grad_value(bcs)[...,1]
    b12 = np.einsum('i,ijk,ij,j -> jk',ws,phi,s12,cellmeasure)
    b2 = np.zeros(gdof)
    np.add.at(b2,cell2dof,b12)

    x1 = spsolve(A,b1)
    x2 = spsolve(A,b2)

    grads = space.function(dim=2)
    grads[:,0] = x1[:]
    grads[:,1] = x2[:]
    return grads

def re(s,space,grads):
    print("开始重置")
    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    
    grads = grads(bcs)
    n = np.sqrt(grads[...,0]**2 + grads[...,1]**2)[...,np.newaxis]
    tag = grads==0
    n = grads/n
    n[tag] = 0
    error = 100
    
    A00 = np.einsum('i,ijk,ijm,j -> jkm',\
            ws,phi,phi,cellmeasure)
    I = np.broadcast_to(cell2dof[:,:,None],shape = A00.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape = A00.shape)
    A0 = csr_matrix((A00.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    
    A11 = np.einsum('i,ijk,ij,ijm,j -> jkm',\
            ws,gphi[...,0],n[...,0],phi,cellmeasure)
    A12 = np.einsum('i,ijk,ij,ijm,j -> jkm',\
            ws,gphi[...,1],n[...,1],phi,cellmeasure)
    I = np.broadcast_to(cell2dof[:,:,None],shape = A11.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape = A11.shape)
    A1 = A11+A12
    A1 = csr_matrix((A1.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    
    A21 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
            ws,gphi[...,0],n[...,0],gphi[...,0],n[...,0],cellmeasure)
    A22 = np.einsum('i,ijk,ij,ijm,ij,j-> jkm',\
            ws,gphi[...,1],n[...,1],gphi[...,1],n[...,1],cellmeasure)
    A23 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
            ws,gphi[...,1],n[...,1],gphi[...,0],n[...,0],cellmeasure)
    A24 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
            ws,gphi[...,0],n[...,0],gphi[...,1],n[...,1],cellmeasure)
    A2 = A21+A22+A23+A24
    A2 = csr_matrix((A2.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    while True:         
        A31 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
                ws,gphi[...,0],n[...,0],phi,s(bcs),cellmeasure)
        A32 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
                ws,gphi[...,1],n[...,1],phi,s(bcs),cellmeasure)
        A3 = A31+A32
        A3 = csr_matrix((A3.flat,(I.flat,J.flat)),shape=(gdof,gdof))
        
        A = A0 - (dt/2)*A1 + (epsilon*dt/2)*A2 + dt*A3
         
        b = A0@s + (dt/2)*A1*s - (epsilon*dt/2)*A2@s

        x = spsolve(A,b)

        errorold = error
        sold = s
        error = space.function()
        error[:]=x[:]-s[:]
        error = integralalg.L2_norm(error)/dt
        s[:] = x[:]
        print(error)
        if error>=errorold or error <=0.01 :
            return s
            break
def area(s,space):
    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    
    fun = space.function()
    fun[:] = 1/2*(np.abs(0.5-s[:])/(0.5-s[:])+1)
    value = space.integralalg.integral(fun)
    return value
    
    

fname = './test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['uh'] = s0
mesh.to_vtk(fname=fname)

for i in range(0,nt):
    t1 = timeline.next_time_level()
    if np.abs(t1- T/2)<1e-10:
        u[:] = -u
    print("t1=",t1)
    print("面积:",area(s0,space))
    s0 = SUPG(u,s0,space)
    gra = grads(s0,space)
    #s0 = re(s0,space,gra)
    mesh.nodedata['uh'] = s0
    mesh.nodedata['u'] = u
    #mesh.celldata['grad'] = norm_s
    fname = './test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.to_vtk(fname=fname)
    timeline.advance()

