import argparse
import sys
import numpy as np
import matplotlib

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from mumps import DMumpsContext

import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.boundarycondition import DirichletBC 
from fealpy.timeintegratoralg import UniformTimeLine
## Stokes model
from navier_stokes_mold_2d import Poisuille as PDE

## error anlysis tool
from fealpy.tools import showmultirate

# 参数设置
udegree = 2
pdegree = 1
fdegree = 1
nx = 50
ny = 50
nt = 500
T = 5
rho = 1.0
mu=1.0
udim = 2
epsilon = (0.02**0.9)/2

def pic(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.15)**2+(y-0.5)**2)-0.15
    val = 1/(1 + np.exp(-val/epsilon))
    return val


pde = PDE()
smesh = MF.boxmesh2d(pde.domain(), nx=nx, ny=ny, meshtype='tri')
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt

uspace = LagrangeFiniteElementSpace(smesh,p=udegree)
pspace = LagrangeFiniteElementSpace(smesh,p=pdegree)
fspace = LagrangeFiniteElementSpace(smesh,p=fdegree)


u0 = uspace.function(dim=udim)
us = uspace.function(dim=udim)
u1 = uspace.function(dim=udim)

p0 = pspace.function()
p1 = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = pgdof+2*ugdof

##矩阵组装准备
qf = smesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = smesh.entity_measure('cell')

## 速度空间
uphi = uspace.basis(bcs)
ugphi = uspace.grad_basis(bcs)
ucell2dof = uspace.cell_to_dof()
NC = smesh.number_of_cells()

epqf = uspace.integralalg.edgeintegrator
epbcs,epws = epqf.get_quadrature_points_and_weights()

## 压力空间
pphi = pspace.basis(bcs)
pgphi = pspace.grad_basis(bcs)
pcell2dof = pspace.cell_to_dof()

index = smesh.ds.boundary_face_index()
ebc = smesh.entity_barycenter('face',index=index)
flag = pde.is_p_boundary(ebc)
index = index[flag]# p边界条件的index

emeasure = smesh.entity_measure('face',index=index)
face2dof = uspace.face_to_dof()[index]
n = smesh.face_unit_normal(index=index)


## 界面
s0 = fspace.interpolation(pic)

def SUPG(u,s0,space):
    print("开始界面")
    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    s1 = space.function() 
    
    norm_u = np.sum(space.integralalg.L2_norm(u,celltype=True),axis=1)
    h = smesh.entity_measure('cell')
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

def integral_matrix():
    E0 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],ugphi[...,0],cellmeasure)
    E1 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],ugphi[...,1],cellmeasure)
    E2 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],ugphi[...,1],cellmeasure)
    E3 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],ugphi[...,0],cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = E0.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = E0.shape)
    E00 = csr_matrix(((E0+1/2*E1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E11 = csr_matrix(((E1+1/2*E0).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E10 = csr_matrix(((1/2*E2).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E01 = csr_matrix(((1/2*E3).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    E = vstack([hstack([E00,E01]),hstack([E10,E11])])
    return E

def edge_matrix(pfun,gfun,nfun): 
    n = nfun(index=index)

    edge2cell = smesh.ds.edge2cell[index]
    egphi = gfun(epbcs,edge2cell[:,0],edge2cell[:,2])
    ephi = pfun(epbcs)
    
    pgx0 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,0],n[:,0],emeasure)
    pgy1 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,1],n[:,1],emeasure)
    pgx1 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,0],n[:,1],emeasure)
    pgy0 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,1],n[:,0],emeasure)

    J1 = np.broadcast_to(face2dof[:,:,None],shape =pgx0.shape) 
    tag = edge2cell[:,0]
    I1 = np.broadcast_to(ucell2dof[tag][:,None,:],shape = pgx0.shape)
    
    D00 = csr_matrix((pgx0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D11 = csr_matrix((pgy1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D01 = csr_matrix((pgy0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D10 = csr_matrix((pgx1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))

    matrix = vstack([hstack([D00,D10]),hstack([D01,D11])])
        
    return matrix

#组装第一个方程的左端矩阵
H = uspace.mass_matrix()
H = bmat([[H,None],[None,H]],format='csr')
E = integral_matrix()
D = edge_matrix(uspace.face_basis,uspace.edge_grad_basis,smesh.face_unit_normal)
A = (rho/dt)*H+mu*E -1/2*D

##边界处理
isBDof = uspace.boundary_dof(threshold=pde.is_wall_boundary)
isBDof = np.hstack((isBDof,isBDof))
bdIdx = np.zeros((A.shape[0],),np.int)
bdIdx[isBDof] = 1
Tbd = spdiags(bdIdx,0,A.shape[0],A.shape[0])
T = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
A = T@A + Tbd

#组装第二个方程的左端矩阵
B1 = pspace.stiff_matrix()
isBDof = pspace.boundary_dof(threshold=pde.is_p_boundary)
bdIdx = np.zeros((B1.shape[0],),np.int)
bdIdx[isBDof] = 1
Tbd = spdiags(bdIdx,0,B1.shape[0],B1.shape[0])
T = spdiags(1-bdIdx,0,B1.shape[0],B1.shape[0])
B =  T@B1 + Tbd

#组装第三个方程的左端矩阵
C = uspace.mass_matrix()
C = bmat([[C,None],[None,C]],format='csr')

fname = './channel_'+ str(0).zfill(10) + '.vtu'
smesh.nodedata['s'] = s0
smesh.to_vtk(fname=fname)

ctx = DMumpsContext()
ctx.set_silent()
errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(0,nt):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    #组装第一个方程的右端向量
    fuu = u0(bcs)
    fbb1 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,fuu,cellmeasure)
    fb1 = np.zeros((ugdof,2))
    np.add.at(fb1,(ucell2dof,np.s_[:]),fbb1)
    fb1 = (rho/dt)*fb1

    fgu = u0.grad_value(bcs)
    fbb2 = np.einsum('i,ijn,ijk,ijmk,j -> jnm',ws,uphi,fuu,fgu,cellmeasure)
    fb2 = np.zeros((ugdof,2))
    np.add.at(fb2,(ucell2dof,np.s_[:]),fbb2)
    
    fb3 = E@u0.flatten(order='F')
    
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
    
    fb6 = ((mu/2))*D@u0.flatten(order='F') 

    b1 = (fb1 - rho*fb2-fb5).flatten(order='F')
    b1 = b1+fb4-mu*fb3+fb6
    
    fisbdf = uspace.boundary_dof(threshold = pde.is_wall_boundary)
    ipoint = uspace.interpolation_points()
    ue2 = pde.velocity(ipoint)
    b1[0:len(fisbdf)][fisbdf] = ue2[fisbdf][:,0]
    b1[len(fisbdf):][fisbdf] = ue2[fisbdf][:,1]
    
    ctx.set_centralized_sparse(A)
    x = b1.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    
    us[:,0] = x[0:ugdof]
    us[:,1] = x[ugdof:]
    #组装第二个方程的右端向量
         
    b21 = pspace.source_vector(us.div_value)
    b21 = -b21/dt
    b22 = B1@p0
    b2 = b21+b22
    ispBDof = pspace.is_boundary_dof(threshold=pde.is_p_boundary)
    ipoint = pspace.interpolation_points()
    pe2 = pde.pressure(ipoint)
    b2[ispBDof] = pe2[ispBDof]

    ctx.set_centralized_sparse(B)
    x = b2.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    p1[:] = x[:]

    #组装第三个方程的右端向量
    tb1 = C@us.flatten(order='F')
    gp = p1.grad_value(bcs)-p0.grad_value(bcs)
    tbb2 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,gp,cellmeasure)
    tb2 = np.zeros((ugdof,2))
    np.add.at(tb2,(ucell2dof,np.s_[:]),tbb2)
    b3 = tb1 - dt*(tb2.flatten(order='F')) 
    
    ctx.set_centralized_sparse(C)
    x = b3.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    u1[:,0] = x[0:ugdof]
    u1[:,1] = x[ugdof:]
    
    s0 = SUPG(u1,s0,fspace)
    #gra = grads(s0,fspace)
    #s0 = re(s0,space,gra)
    smesh.nodedata['s'] = s0
    smesh.nodedata['u'] = u1
    
    fname = './channel_'+ str(i+1).zfill(10) + '.vtu'
    smesh.to_vtk(fname=fname)
    
    uc1 = pde.velocity(smesh.node)
    NN = smesh.number_of_nodes()
    uc2 = u1[:NN]
    up1 = pde.pressure(smesh.node)
    up2 = p1[:NN]
   
    errorMatrix[0,i] = uspace.integralalg.L2_error(pde.velocity,u1)
    errorMatrix[1,i] = pspace.integralalg.error(pde.pressure,p1)
    errorMatrix[2,i] = np.abs(uc1-uc2).max()
    errorMatrix[3,i] = np.abs(up1-up2).max()
    print(errorMatrix[:,i])
    u0[:] = u1 
    p0[:] = p1
    # 时间步进一层 
    tmesh.advance()

ctx.destroy()
print("uL2:",errorMatrix[2,-1])
print("pL2:",errorMatrix[1,-1])
print("umax:",errorMatrix[2,-1])
print("pmax:",errorMatrix[3,-1])
'''
fig1 = plt.figure()
node = smesh.node
x = tuple(node[:,0])
y = tuple(node[:,1])
NN = smesh.number_of_nodes()
u = u1[:NN]
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
