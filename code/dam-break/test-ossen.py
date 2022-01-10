#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: test.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年11月26日 星期五 17时25分01秒
'''
import numpy as np
import matplotlib.pyplot as plt

from fealpy.decorator import cartesian,barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.timeintegratoralg import UniformTimeLine

from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from scipy.sparse.linalg import spsolve
from mumps import DMumpsContext

eps = 1e-12
domain = [0,0.0584,0,0.0438]
dam_domain=[0,0.0146,0,0.0292]
#a = 0.05715
#domain = [0,5*a,0,5*a]
#dam_domain=[0,a,0,a]
rho_water = 1000  #kg/m^3
rho_air = 1  #kg/m^3
mu_water = 0.001 #pa*s
mu_air = 0.00001 #Pa*s

@cartesian
def is_down_boundary(p):
    return (np.abs(p[..., 1]) < eps)
@cartesian
def is_up_boundary(p):
    return (np.abs(p[..., 1] - domain[3]) < eps)
@cartesian
def is_ud_boundary(p):
    return (np.abs(p[..., 1]) < eps)| (np.abs(p[..., 1])-domain[3] < eps) 
@cartesian
def is_rl_boundary(p):
    return (np.abs(p[..., 0]) < eps) |(np.abs(p[..., 0]-domain[1]) < eps)
@cartesian 
def source(p):
    x = p[...,0]
    y = p[...,1]
    g = 20 #m/s^2
    value = np.zeros(p.shape)
    value[...,1] = -g
    return value

udegree = 2
pdegree = 1
ns = 70
nt= 2000
T = 1


mesh = MF.boxmesh2d(domain,nx=ns,ny=ns,meshtype='tri')
timeline = UniformTimeLine(0,T,nt)

uspace = LagrangeFiniteElementSpace(mesh,p=udegree)
pspace = LagrangeFiniteElementSpace(mesh,p=pdegree)

u0 = uspace.function(dim=2)
us = uspace.function(dim=2)
u1 = uspace.function(dim=2)

p0 = pspace.function()
p1 = pspace.function()


#矩阵组装准备
dt = timeline.dt
qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')
ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = pgdof+2*ugdof
## 速度
uphi = uspace.basis(bcs)
ugphi = uspace.grad_basis(bcs)
ucell2dof = uspace.cell_to_dof()
## 压力
pphi = pspace.basis(bcs)
pgphi = pspace.grad_basis(bcs)
pcell2dof = pspace.cell_to_dof()
#边
epqf = uspace.integralalg.edgeintegrator
epbcs,epws = epqf.get_quadrature_points_and_weights()
index = mesh.ds.boundary_face_index()
ebc = mesh.entity_barycenter('face',index=index)
flag = is_up_boundary(ebc)
index = index[flag]# p边界条件的index
emeasure = mesh.entity_measure('face',index=index)
face2dof = uspace.face_to_dof()[index]
n = mesh.face_unit_normal(index=index)

@cartesian
def dist(p):
    x = p[...,0]
    y = p[...,1]
    x0 = dam_domain[0]
    x1 = dam_domain[1]
    y0 = dam_domain[2]
    y1 = dam_domain[3]
    value = np.zeros(shape=x.shape)
    tagx =  np.logical_and(x0<=x,x<=x1)
    tagy =  np.logical_and(y0<=y,y<=y1)
    area00 = np.logical_and(tagx,tagy)
    area01 = np.logical_and(tagx,y>y1)
    area10 = np.logical_and(x>x1,tagy)
    area11 = np.logical_and(x>x1,y>y1)
    value[area00] = -np.minimum(x1-x,y1-y)[area00]
    value[area01] = (y-y1)[area01]
    value[area10] = (x-x1)[area10]
    value[area11] = np.sqrt((x-x1)**2+(y-y1)**2)[area11]
    return value

def integral_matrix(r,s0):
    E0 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs,s0),ugphi[...,0],ugphi[...,0],cellmeasure)
    E1 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs,s0),ugphi[...,1],ugphi[...,1],cellmeasure)
    E2 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs,s0),ugphi[...,0],ugphi[...,1],cellmeasure)
    E3 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs,s0),ugphi[...,1],ugphi[...,0],cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = E0.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = E0.shape)
    E00 = csr_matrix(((E0+1/2*E1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E11 = csr_matrix(((E1+1/2*E0).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E10 = csr_matrix(((1/2*E2).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E01 = csr_matrix(((1/2*E3).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E = bmat([[E00,E01],[E10,E11]])
    return E

@barycentric
def rho(bcs,s0):
    tag_a = s0(bcs)>0
    tag_w = s0(bcs)<=0
    value = np.zeros(shape=s0(bcs).shape)
    value[tag_a] = rho_air
    value[tag_w] = rho_water
    return value
@barycentric
def mu(bcs,s0):
    tag_a = s0(bcs)>0
    tag_w = s0(bcs)<=0
    value = np.zeros(shape=s0(bcs).shape)
    value[tag_a] = mu_air
    value[tag_w] = mu_water
    return value

f = uspace.interpolation(source)
s0 = pspace.interpolation(dist)

mesh.nodedata['dist'] = s0
#mesh.nodedata['rho'] = rho
#mesh.nodedata['mu'] = mu
mesh.nodedata['p'] = p0
mesh.nodedata['u'] = u0
mesh.nodedata['f'] = f

fname = './ossen_'+ str(0).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)

ctx = DMumpsContext()
ctx.set_silent()
for i in range(0,nt):
    t1 = timeline.next_time_level()
    print("t1=",t1)

    #第一个方程右端  
    a1 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,rho(bcs,s0),uphi,uphi,cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = a1.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = a1.shape)
    A1 = 1/dt*csr_matrix((a1.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    
    
    B1,B2 = uspace.div_matrix(pspace)
    
    a2 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,mu(bcs,s0),ugphi[...,0],ugphi[...,0],cellmeasure)
    a3 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,mu(bcs,s0),ugphi[...,1],ugphi[...,1],cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = a2.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = a2.shape)
    A2 = csr_matrix(((a2+a3).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A2 = integral_matrix(mu,s0)
    
    a4 = np.einsum('i,ij,ijk,ijn,ijmn,j->jkm',ws,rho(bcs,s0),uphi,u0(bcs),ugphi,cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = a4.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = a4.shape)
    A3 = csr_matrix((a4.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
 
    #A = bmat([[A1+A2+A3,None,-B1],[None,A1+A2+A3,-B2],[-B1.T,-B2.T,None]],format='csr')
    
    A = bmat([[A1+A3,None],[None,A1+A3]],format='csr')
    A =  A + A2
    AA = bmat([[-B1],[-B2]],format='csr')
    A = bmat([[A,AA],[AA.T,None]],format='csr')
    
    ##边界处理
    isUDBDof = uspace.boundary_dof(threshold=is_ud_boundary)
    isRLBDof = uspace.boundary_dof(threshold=is_rl_boundary)
    isUPBDof = pspace.boundary_dof(threshold=is_up_boundary)
    isBDof = np.hstack([isRLBDof,isUDBDof,isUPBDof])
    bdIdx = np.zeros((A.shape[0],),np.int)
    bdIdx[isBDof] = 1
    Tbd = spdiags(bdIdx,0,A.shape[0],A.shape[0])
    T = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
    A = T@A + Tbd
    
    
    #第一个方程右端
    fu = f(bcs)
    fbb0 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,rho(bcs,s0),uphi,fu,cellmeasure)
    fb0 = np.zeros((ugdof,2))
    np.add.at(fb0,(ucell2dof,np.s_[:]),fbb0)
    
    b1 = (fb0 + A1@u0).flatten(order='F')
    b = np.hstack([b1, np.zeros(pgdof)])
    
    b[isBDof] = 0
    ctx.set_centralized_sparse(A)
    x = b.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    
    u1[:,0] = x[0:ugdof]
    u1[:,1] = x[ugdof:2*ugdof]
    p1[:] =x[2*ugdof:] 
    
    ## 计算界面
    d0 = np.einsum('i,ijk,ijm,ij,j -> jkm',ws,pphi,pgphi[...,0],u1(bcs)[...,0],cellmeasure)
    d1 = np.einsum('i,ijk,ijm,ij,j -> jkm',ws,pphi,pgphi[...,1],u1(bcs)[...,1],cellmeasure)
    d = d0+d1
    I = np.broadcast_to(pcell2dof[:,:,None],shape = d.shape)
    J = np.broadcast_to(pcell2dof[:,None,:],shape = d.shape)
    D1 = pspace.mass_matrix()
    D2 = csr_matrix((d.flat,(I.flat,J.flat)),shape=(pgdof,pgdof))

    D = D1+dt/2*D2
    
    b4 = D@s0-dt/2*D2@s0
    ctx.set_centralized_sparse(D)
    x = b4.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
   
    s0[:] = x[:]
    print(uspace.integralalg.L2_norm(s0.grad_value))
     
    fname = './ossen_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['dist'] = s0
    mesh.nodedata['u'] = u1
    mesh.nodedata['p'] = p1
    #mesh.nodedata['mu'] = mu
    #mesh.nodedata['rho'] = rho
    mesh.nodedata['f'] = f
    mesh.to_vtk(fname=fname)
    timeline.advance()

ctx.destroy()

