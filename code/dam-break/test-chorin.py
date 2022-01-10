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

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.timeintegratoralg import UniformTimeLine

from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from scipy.sparse.linalg import spsolve
from mumps import DMumpsContext

eps = 1e-12
domain = [0,0.584,0,0.438]
dam_domain=[0,0.146,0,0.292]
rho_water = 1000  #kg/m^3
rho_air = 1  #kg/m^3
mu_water = 1e-3 #pa*s
mu_air = 1e-5 #Pa*s

@cartesian
def is_p_boundary(p):
    return (np.abs(p[..., 1] - domain[3]) < eps)
  
@cartesian
def is_wall_boundary(p):
    return (np.abs(p[..., 0]) < eps) | (np.abs(p[..., 1]) < eps) | \
           (np.abs(p[..., 0]-domain[1]) < eps)
 

def source(p):
    x = p[...,0]
    y = p[...,1]
    g = 9.81 #m/s^2
    value = np.zeros(p.shape)
    value[...,1] = -g
    return value


udegree = 1
pdegree = 1
ns = 50
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
uldof = uspace.number_of_local_dofs()
pldof = pspace.number_of_local_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = pgdof+2*ugdof
## 速度
uphi = uspace.basis(bcs)
ugphi = uspace.grad_basis(bcs)
ucell2dof = uspace.cell_to_dof()
## 压力
pphi = pspace.basis(bcs)
pgphi = pspace.grad_basis(bcs)
index = mesh.ds.boundary_face_index()
ebc = mesh.entity_barycenter('face',index=index)
flag = is_p_boundary(ebc)
index = index[flag]
pcell2dof = pspace.cell_to_dof()
#边
epqf = uspace.integralalg.edgeintegrator
epbcs,epws = epqf.get_quadrature_points_and_weights()
index = mesh.ds.boundary_face_index()
ebc = mesh.entity_barycenter('face',index=index)
flag = is_p_boundary(ebc)
index = index[flag]# p边界条件的index
emeasure = mesh.entity_measure('face',index=index)
face2dof = uspace.face_to_dof()[index]
n = mesh.face_unit_normal(index=index)


def edge_matrix(pfun,gfun,nfun,r): 
    n = nfun(index=index)

    edge2cell = mesh.ds.edge2cell[index]
    egphi = gfun(epbcs,edge2cell[:,0],edge2cell[:,2])
    ephi = pfun(epbcs)
    pgx0 = np.einsum('i,ij,ijk,jim,j,j->jkm',epws,r(epbcs)[:,index],ephi,egphi[...,0],n[:,0],emeasure)
    pgy1 = np.einsum('i,ij,ijk,jim,j,j->jkm',epws,r(epbcs)[:,index],ephi,egphi[...,1],n[:,1],emeasure)
    pgx1 = np.einsum('i,ij,ijk,jim,j,j->jkm',epws,r(epbcs)[:,index],ephi,egphi[...,0],n[:,1],emeasure)
    pgy0 = np.einsum('i,ij,ijk,jim,j,j->jkm',epws,r(epbcs)[:,index],ephi,egphi[...,1],n[:,0],emeasure)

    J1 = np.broadcast_to(face2dof[:,:,None],shape =pgx0.shape) 
    tag = edge2cell[:,0]
    I1 = np.broadcast_to(ucell2dof[tag][:,None,:],shape = pgx0.shape)
    
    D00 = csr_matrix((pgx0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D11 = csr_matrix((pgy1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D01 = csr_matrix((pgy0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D10 = csr_matrix((pgx1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))

    matrix = vstack([hstack([D00,D10]),hstack([D01,D11])])
        
    return matrix

def integral_matrix(r):
    E0 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs),ugphi[...,0],ugphi[...,0],cellmeasure)
    E1 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs),ugphi[...,1],ugphi[...,1],cellmeasure)
    E2 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs),ugphi[...,0],ugphi[...,1],cellmeasure)
    E3 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs),ugphi[...,1],ugphi[...,0],cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = E0.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = E0.shape)
    E00 = csr_matrix(((E0+1/2*E1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E11 = csr_matrix(((E1+1/2*E0).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E10 = csr_matrix(((1/2*E2).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E01 = csr_matrix(((1/2*E3).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E = bmat([[E00,E01],[E10,E11]])
    return E
def integral_matrix(r):
    E0 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs),ugphi[...,0],ugphi[...,0],cellmeasure)
    E1 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs),ugphi[...,1],ugphi[...,1],cellmeasure)
    E2 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs),ugphi[...,0],ugphi[...,1],cellmeasure)
    E3 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,r(bcs),ugphi[...,1],ugphi[...,0],cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = E0.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = E0.shape)
    E00 = csr_matrix(((E0+1/2*E1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E11 = csr_matrix(((E1+1/2*E0).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E10 = csr_matrix(((1/2*E2).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E01 = csr_matrix(((1/2*E3).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E = bmat([[E00,E01],[E10,E11]])
    return E

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
    value[area00] = np.maximum(x-x1,y-y1)[area00]
    value[area01] = (y-y1)[area01]
    value[area10] = (x-x1)[area10]
    value[area11] = np.sqrt((x-x1)**2+(y-y1)**2)[area11]
    return value
'''
def mu(p):
    tag_w = dist(p)<0
    tag_a = dist(p)>-0
    value = np.zeros(shape=p[...,0].shape)
    value[tag_w] = mu_water
    value[tag_a] = mu_air 
    return value

def rho(p):
    tag_w = dist(p)<0
    tag_a = dist(p)>-0
    value = np.zeros(shape=p[...,0].shape)
    value[tag_w] = rho_water
    value[tag_a] = rho_air 
    return value
'''
rho = uspace.function()
mu = uspace.function()
f = uspace.interpolation(source)
s0 = uspace.interpolation(dist)

tag_w = s0<0
tag_a = s0>=0
mu[tag_w] = mu_water
mu[tag_a] = mu_air
rho[tag_w] = rho_water
rho[tag_a] = rho_air

mesh.nodedata['dist'] = s0
mesh.nodedata['rho'] = rho
mesh.nodedata['mu'] = mu
mesh.nodedata['p'] = p0
mesh.nodedata['u'] = u0
mesh.nodedata['f'] = f

fname = './chorin_'+ str(0).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)

ctx = DMumpsContext()
ctx.set_silent()
for i in range(0,nt):
    t1 = timeline.next_time_level()
    print("t1=",t1)

    #第一个方程右端  
    a1 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,rho(bcs),uphi,uphi,cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = a1.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = a1.shape)
    A1 = csr_matrix((a1.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A1 = bmat([[A1,None],[None,A1]],format='csr')
    
    a2 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,mu(bcs),ugphi[...,0],ugphi[...,0],cellmeasure)
    a3 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,mu(bcs),ugphi[...,1],ugphi[...,1],cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = a2.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = a2.shape)
    A2 = csr_matrix(((a2+a3).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A2 = bmat([[A2,None],[None,A2]],format='csr')
    A = 1/dt*A1 + A2 
    ##边界处理
    isBDof = uspace.boundary_dof(threshold=is_wall_boundary)
    isBDof = np.hstack((isBDof,isBDof))
    bdIdx = np.zeros((A.shape[0],),np.int)
    bdIdx[isBDof] = 1
    Tbd = spdiags(bdIdx,0,A.shape[0],A.shape[0])
    T = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
    A = T@A + Tbd
    
    #第二个矩阵左端
    
    b2 = np.einsum('i,ijk,ijm,j->jkm',ws,pgphi[...,0],pgphi[...,0],cellmeasure)
    b3 = np.einsum('i,ijk,ijm,j->jkm',ws,pgphi[...,1],pgphi[...,1],cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = b2.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = b2.shape)
    B1 = csr_matrix(((b2+b3).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    
    #B1 = pspace.stiff_matrix()
    isBDof = pspace.boundary_dof(is_p_boundary)
    bdIdx = np.zeros((B1.shape[0],),np.int)
    bdIdx[isBDof] = 1
    Tbd = spdiags(bdIdx,0,B1.shape[0],B1.shape[0])
    T = spdiags(1-bdIdx,0,B1.shape[0],B1.shape[0])
    B =  T@B1 + Tbd
    #B = B1
    
    #第三个矩阵左端
    C = 1/dt*A1
    
    #第一个方程右端
    fu = f(bcs)
    fbb0 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,rho(bcs),uphi,fu,cellmeasure)
    fb0 = np.zeros((ugdof,2))
    np.add.at(fb0,(ucell2dof,np.s_[:]),fbb0)

    fuu = u0(bcs)
    fbb1 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,rho(bcs),uphi,fuu,cellmeasure)
    fb1 = np.zeros((ugdof,2))
    np.add.at(fb1,(ucell2dof,np.s_[:]),fbb1)
    fb1 = fb1

    fgu = u0.grad_value(bcs)
    fbb2 = np.einsum('i,ij,ijk,ijn,ijmn,j -> jkm',ws,rho(bcs),uphi,fuu,fgu,cellmeasure)
    fb2 = np.zeros((ugdof,2))
    np.add.at(fb2,(ucell2dof,np.s_[:]),fbb2)
     
    b1 = (fb0 + 1/dt*fb1 - fb2).flatten(order='F')
    
    fisbdf = uspace.boundary_dof(threshold = is_wall_boundary)
    b1[0:len(fisbdf)][fisbdf] = 0
    b1[len(fisbdf):][fisbdf] = 0
    
    ctx.set_centralized_sparse(A)
    x = b1.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    
    us[:,0] = x[0:ugdof]
    us[:,1] = x[ugdof:]
    #组装第二个方程的右端向量
    
    fgus = us.grad_value(bcs).trace(axis1=-2,axis2=-1)
    fbb21 = np.einsum('i,ij,ijk,ij,j -> jk',ws,rho(bcs),pphi,fgus,cellmeasure)
    b21 = np.zeros(pgdof)
    np.add.at(b21,(pcell2dof),fbb21)
    
    b2 = -1/dt*b21
    ispBDof = pspace.is_boundary_dof(is_p_boundary)
    b2[ispBDof] = 0
    
    ctx.set_centralized_sparse(B)
    x = b2.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    p1[:] = x[:]

    #组装第三个方程的右端向量
    tb1 = A1@us.flatten(order='F')
    gp = p1.grad_value(bcs)
    tbb2 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,gp,cellmeasure)
    tb2 = np.zeros((ugdof,2))
    np.add.at(tb2,(ucell2dof,np.s_[:]),tbb2)
    b3 = 1/dt*tb1 - tb2.flatten(order='F') 
    
    ctx.set_centralized_sparse(C)
    x = b3.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    u1[:,0] = x[0:ugdof]
    u1[:,1] = x[ugdof:]
    
    ## 计算界面
    d0 = np.einsum('i,ijk,ijm,ij,j -> jkm',ws,uphi,ugphi[...,0],u1(bcs)[...,0],cellmeasure)
    d1 = np.einsum('i,ijk,ijm,ij,j -> jkm',ws,uphi,ugphi[...,1],u1(bcs)[...,1],cellmeasure)
    d = d0+d1
    I = np.broadcast_to(ucell2dof[:,:,None],shape = d.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = d.shape)
    D1 = uspace.mass_matrix()
    D2 = csr_matrix((d.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))

    D = D1+dt/2*D2
    
    b4 = D@s0-dt/2*D2@s0
    ctx.set_centralized_sparse(D)
    x = b4.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
   
    s0[:] = x[:]
    
    tag_w = s0<0
    tag_a = s0>=0
    mu[tag_w] = mu_water
    mu[tag_a] = mu_air
    rho[tag_w] = rho_water
    rho[tag_a] = rho_air
    
    fname = './chorin_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['dist'] = s0
    mesh.nodedata['u'] = u1
    mesh.nodedata['p'] = p1
    mesh.nodedata['mu'] = mu
    mesh.nodedata['rho'] = rho
    mesh.nodedata['f'] = f
    mesh.to_vtk(fname=fname)
    timeline.advance()

ctx.destroy()

