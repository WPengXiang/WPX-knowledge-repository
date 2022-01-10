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
import matplotlib
from fealpy.decorator import cartesian,barycentric
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
'''
a = 0.05715
domain = [0,5*a,0,4*a]
dam_domain=[0,a,0,2*a]
'''
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
def is_u_boundary(p):
    return (np.abs(p[..., 0]) < eps) |(np.abs(p[..., 0]-domain[1]) < eps)\
            |(np.abs(p[..., 1]) < eps)
@cartesian
def is_rl_boundary(p):
    return (np.abs(p[..., 0]) < eps) |(np.abs(p[..., 0]-domain[1]) < eps)
@cartesian 
def source(p):
    x = p[...,0]
    y = p[...,1]
    g = 9.81 #m/s^2
    value = np.zeros(p.shape)
    value[...,1] = -g
    return value


udegree = 2
pdegree = 1
ns = 100
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
integralalg = pspace.integralalg
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
index = index[flag]# 上边界条件的index
emeasure = mesh.entity_measure('face',index=index)
face2dof = uspace.face_to_dof()[index]
n = mesh.face_unit_normal(index=index)


def edge_matrix(pfun,gfun,nfun,r,s0): 
    n = nfun(index=index)
    edge2cell = mesh.ds.edge2cell[index]
    egphi = gfun(epbcs,edge2cell[:,0],edge2cell[:,2])
    ephi = pfun(epbcs)
    pgx0 = np.einsum('i,ij,ijk,jim,j,j->jkm',epws,r(epbcs,s0)[:,index],ephi,egphi[...,0],n[:,0],emeasure)
    pgy1 = np.einsum('i,ij,ijk,jim,j,j->jkm',epws,r(epbcs,s0)[:,index],ephi,egphi[...,1],n[:,1],emeasure)
    pgx1 = np.einsum('i,ij,ijk,jim,j,j->jkm',epws,r(epbcs,s0)[:,index],ephi,egphi[...,0],n[:,1],emeasure)
    pgy0 = np.einsum('i,ij,ijk,jim,j,j->jkm',epws,r(epbcs,s0)[:,index],ephi,egphi[...,1],n[:,0],emeasure)

    J1 = np.broadcast_to(face2dof[:,:,None],shape =pgx0.shape) 
    tag = edge2cell[:,0]
    I1 = np.broadcast_to(ucell2dof[tag][:,None,:],shape = pgx0.shape)
    
    D00 = csr_matrix((pgx0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D11 = csr_matrix((pgy1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D01 = csr_matrix((pgy0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D10 = csr_matrix((pgx1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))

    matrix = vstack([hstack([D00,D10]),hstack([D01,D11])])    
    return matrix

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
    E = vstack([hstack([E00,E01]),hstack([E10,E11])])
    return E


@cartesian
def dist(p):
    x = p[...,0]
    y = p[...,1]
    x1 = dam_domain[1]
    y1 = dam_domain[3]
    value = np.zeros(shape=x.shape)
    area00 = np.logical_and(x<=x1,y<=y1)
    area01 = np.logical_and(x<=x1,y>y1)
    area10 = np.logical_and(x>x1,y<=y1)
    area11 = np.logical_and(x>x1,y>y1)
    value[area00] = np.maximum(x-x1,y-y1)[area00]
    value[area01] = (y-y1)[area01]
    value[area10] = (x-x1)[area10]
    value[area11] = np.sqrt((x-x1)**2+(y-y1)**2)[area11]
    return value

@barycentric
def rho(bcs,s0):
    tag_a = s0(bcs)>=0
    tag_w = s0(bcs)<0
    value = np.zeros(shape=s0(bcs).shape)
    value[tag_a] = rho_air
    value[tag_w] = rho_water
    return value

@barycentric
def mu(bcs,s0):
    tag_a = s0(bcs)>=0
    tag_w = s0(bcs)<0
    value = np.zeros(shape=s0(bcs).shape)
    value[tag_a] = mu_air
    value[tag_w] = mu_water
    return value

f = uspace.interpolation(source)
s0 = pspace.interpolation(dist)

mesh.nodedata['dist'] = s0
mesh.nodedata['p'] = p0
mesh.nodedata['u'] = u0
mesh.nodedata['f'] = f

fname = './ipcs_'+ str(0).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)

ctx = DMumpsContext()
ctx.set_silent()
for i in range(0,1):
    t1 = timeline.next_time_level()
    print("t1=",t1)

    #print(integralalg.L2_norm(s0.grad_value))
    #第一个方程右端  
    a1 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,rho(bcs,s0),uphi,uphi,cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = a1.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = a1.shape)
    A1 = csr_matrix((a1.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    H = bmat([[H,None],[None,H]],format='csr')

    E = integral_matrix(mu,s0)
    D = edge_matrix(uspace.face_basis,\
            uspace.edge_grad_basis,mesh.face_unit_normal,mu,s0)
    A = 1/dt*H + E - 1/2*D
    
    ##边界处理
    isUDBDof = uspace.boundary_dof(threshold=is_ud_boundary)
    isuBDof = uspace.boundary_dof(threshold=is_u_boundary)
    isRLBDof = uspace.boundary_dof(threshold=is_rl_boundary)
    isuBDof = np.hstack((isuBDof,isuBDof))
    #isBDof = np.hstack((uspace.boundary_dof(),uspace.boundary_dof()))
    bdIdx = np.zeros((A.shape[0],),np.int)
    bdIdx[isuBDof] = 1
    Tbd = spdiags(bdIdx,0,A.shape[0],A.shape[0])
    T = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
    A = T@A + Tbd
    
    #第二个矩阵左端
    B1 = pspace.stiff_matrix()
    ispBDof = pspace.boundary_dof(threshold = is_up_boundary)
    bdIdx = np.zeros((B1.shape[0],),np.int)
    bdIdx[ispBDof] = 1
    Tbd = spdiags(bdIdx,0,B1.shape[0],B1.shape[0])
    T = spdiags(1-bdIdx,0,B1.shape[0],B1.shape[0])
    B =  T@B1 + Tbd
   
    
    #第三个矩阵左端
    C = A1
    
    #第一个方程右端
    
    fu = f(bcs)
    fbb0 = np.einsum('i,ij,ijk,ijm,j -> jkm',ws,rho(bcs,s0),uphi,fu,cellmeasure)
    fb0 = np.zeros((ugdof,2))
    np.add.at(fb0,(ucell2dof,np.s_[:]),fbb0)
    
    fuu = u0(bcs)
    fbb1 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,fuu,cellmeasure)
    fb1 = np.zeros((ugdof,2))
    np.add.at(fb1,(ucell2dof,np.s_[:]),fbb1)

    fgu = u0.grad_value(bcs)
    fbb2 = np.einsum('i,ijn,ijk,ijmk,j -> jnm',ws,uphi,fuu,fgu,cellmeasure)
    fb2 = np.zeros((ugdof,2))
    np.add.at(fb2,(ucell2dof,np.s_[:]),fbb2)
    
    fb3 = A2@u0.flatten(order='F')
    
    fp = p0(bcs)
    fbb4 = np.einsum('i,ij,ijk,j->jk',ws,fp,ugphi[...,0],cellmeasure) 
    fbb5 = np.einsum('i,ij,ijk,j->jk',ws,fp,ugphi[...,1],cellmeasure) 
    fb4 = np.zeros((ugdof))
    fb5 = np.zeros((ugdof))
    np.add.at(fb4,ucell2dof,fbb4)
    np.add.at(fb5,ucell2dof,fbb5)
    fb4 = np.hstack((fb4,fb5)) 
   
    ##p边界
    ep = p0(epbcs)[...,index]
    value = np.einsum('ij,jk->ijk',ep,n)
    ephi = uspace.face_basis(epbcs)
    evalue = np.einsum('i,ijk,ijm,j->jkm',epws,ephi,value,emeasure)
    fb5 = np.zeros((ugdof,2))
    np.add.at(fb5,(face2dof,np.s_[:]),evalue)
    
    fb6 = A3@u0.flatten(order='F') 

    b1 = (fb0+1/dt*fb1 - fb2 - dt*fb5).flatten(order='F')
    b1 = b1 + fb4 - fb3 + 1/2*fb6
    
    b1[isuBDof] = 0
    '''
    fisUDbdf = uspace.boundary_dof(threshold = is_ud_boundary)
    fisRLbdf = uspace.boundary_dof(threshold = is_rl_boundary)
    b1[0:ugdof][fisRLbdf] = 0
    b1[ugdof:][fisUDbdf] = 0
    '''
    ctx.set_centralized_sparse(A)
    x = b1.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    
    us[:,0] = x[0:ugdof]
    us[:,1] = x[ugdof:]
    #组装第二个方程的右端向量

    fgus = us.grad_value(bcs).trace(axis1=-2,axis2=-1)
    fbb22 = np.einsum('i,ij,ijk,ij,j -> jk',ws,rho(bcs,s0),pphi,fgus,cellmeasure)
    b22 = np.zeros(pgdof)
    np.add.at(b22,(pcell2dof),fbb22)
    
    b21 = B1@p0
    
    b2 = b21-(1/dt)*b22
    ispBDof = pspace.is_boundary_dof(is_up_boundary)
    b2[ispBDof] = 0
    
    ctx.set_centralized_sparse(B.T)
    x = b2.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    p1[:] = x[:]

    #组装第三个方程的右端向量
    tb1 = C@us.flatten(order='F')
    gp = p1.grad_value(bcs) -p0.grad_value(bcs)
    tbb2 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,gp,cellmeasure)
    tb2 = np.zeros((ugdof,2))
    np.add.at(tb2,(ucell2dof,np.s_[:]),tbb2)
    b3 = tb1 - dt*(tb2.flatten(order='F')) 
    '''
    fisUDbdf = uspace.boundary_dof(threshold = is_ud_boundary)
    fisRLbdf = uspace.boundary_dof(threshold = is_rl_boundary)
    b3[0:ugdof][fisRLbdf] = 0
    b3[ugdof:][fisUDbdf] = 0
    '''

    ctx.set_centralized_sparse(C)
    x = b3.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    u1[:,0] = x[0:ugdof]
    u1[:,1] = x[ugdof:]
    
    mesh.nodedata['u'] = u1
    mesh.nodedata['p'] = p1
    mesh.nodedata['f'] = f
    '''    
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
    #s0[:] = re(s0)[:]
    '''
    fname = './ipcs_'+ str(i+1).zfill(10) + '.vtu'
    #mesh.nodedata['dist'] = s0
    mesh.to_vtk(fname=fname)
    timeline.advance()

ctx.destroy()

