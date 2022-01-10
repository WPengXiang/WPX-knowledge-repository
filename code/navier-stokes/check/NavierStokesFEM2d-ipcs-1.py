import numpy as np
from mumps import DMumpsContext
import scipy
from scipy.linalg import solve

import matplotlib.pyplot as plt
import matplotlib

from  fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric
from fealpy.boundarycondition import DirichletBC
from fealpy.quadrature import GaussLegendreQuadrature
from scipy.sparse import csr_matrix,hstack,vstack,spdiags

from fealpy.tools.show import showmultirate

## 参数解析
udegree = 1
pdegree = 1
dim = 1
ns = 8
nt = 100

eps = 1e-15
T = 10
rho = 1
mu = 1
inp = 8.0
outp = 0.0

@cartesian
def walldirichlet(p):
    return 0
@cartesian
def is_in_flow_boundary(p):
    return np.abs(p[..., 0]) < eps 
@cartesian
def is_out_flow_boundary(p):
    return np.abs(p[..., 0] - 1.0) < eps

@cartesian
def is_p_boundary(p):
    return (np.abs(p[..., 0]) < eps) | (np.abs(p[..., 0] - 1.0) < eps)
  
@cartesian
def is_wall_boundary(p):
    return (np.abs(p[..., 1]) < eps) | (np.abs(p[..., 1] - 1.0) < eps)

@cartesian
def usolution(p):
    x = p[...,0]
    y = p[...,1]
    u = np.zeros(p.shape)
    u[...,0] = 4*y*(1-y)
    return u

@cartesian 
def psolution(p):
    x = p[...,0]
    y = p[...,1]
    return 8*(1-x)

domain = [0, 1, 0, 1]

smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
tmesh = UniformTimeLine(0, T, nt) # 均匀时间剖分

uspace = LagrangeFiniteElementSpace(smesh, p=udegree)
pspace = LagrangeFiniteElementSpace(smesh, p=pdegree)

u0 = uspace.function(dim=2)
us = uspace.function(dim=2)
u1 = uspace.function(dim=2)

p0 = pspace.function()
p1 = pspace.function()

#组装矩阵准备工作
dt = tmesh.dt
qf = smesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = smesh.entity_measure('cell')

## 速度空间
uphi = uspace.basis(bcs)
ugphi = uspace.grad_basis(bcs)
ugdof = uspace.number_of_global_dofs()
ucell2dof = uspace.cell_to_dof()
Nldof = uspace.number_of_local_dofs()
NC = smesh.number_of_cells()
I = np.broadcast_to(ucell2dof[:,:,None],shape = (NC,Nldof,Nldof))
J = np.broadcast_to(ucell2dof[:,None,:],shape = (NC,Nldof,Nldof))

## 压力空间
pphi = pspace.basis(bcs)
pgphi = pspace.grad_basis(bcs)
pgdof = pspace.number_of_global_dofs()
pcell2dof = pspace.cell_to_dof()

#组装第一个方程的左端矩阵

H = uspace.mass_matrix()
H = (1/dt)*csr_matrix(scipy.linalg.block_diag(H.toarray(),H.toarray()))

E0 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],ugphi[...,0],cellmeasure)
E1 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],ugphi[...,1],cellmeasure)
E2 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],ugphi[...,1],cellmeasure)
E3 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],ugphi[...,0],cellmeasure)
E00 = csr_matrix(((E0+1/2*E1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E11 = csr_matrix(((E1+1/2*E0).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E10 = csr_matrix(((1/2*E2).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E01 = csr_matrix(((1/2*E3).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E = (mu/rho)*vstack([hstack([E00,E01]),hstack([E10,E11])])

index = smesh.ds.boundary_face_index()
ebc = smesh.entity_barycenter('face',index=index)
flag = is_p_boundary(ebc)
index = index[flag]# p边界条件的index

face2dof = uspace.face_to_dof()[index]
n = smesh.face_unit_normal(index=index)
emeasure = smesh.entity_measure('face',index=index)
epqf = uspace.integralalg.edgeintegrator
epbcs,epws = epqf.get_quadrature_points_and_weights()

edge2cell = smesh.ds.edge2cell[index]

egphi = uspace.edge_grad_basis(epbcs,edge2cell[:,0],edge2cell[:,2])
ephi = uspace.face_basis(epbcs)

pgx0 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,0],n[:,0],emeasure)
pgy1 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,1],n[:,1],emeasure)
pgx1 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,0],n[:,1],emeasure)
pgy0 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,1],n[:,0],emeasure)

J1 = np.broadcast_to(face2dof[:,:,None],shape =pgx0.shape) 
tag = edge2cell[:,0]
I1 = np.broadcast_to(ucell2dof[tag][:,None,:],shape = pgx0.shape)
'''
D00 = csr_matrix(((pgx0+1/2*pgy1).flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
D11 = csr_matrix(((pgy1+1/2*pgx0).flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
D01 = csr_matrix(((1/2*pgx1).flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
D10 = csr_matrix(((1/2*pgy0).flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
'''
D00 = csr_matrix((pgx0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
D11 = csr_matrix((pgy1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
D01 = csr_matrix((pgy0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
D10 = csr_matrix((pgx1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))

D = 1/2*vstack([hstack([D00,D10]),hstack([D01,D11])])

A = H + E - D


##边界处理
isBDof = uspace.boundary_dof(threshold=is_wall_boundary)
isBDof = np.hstack((isBDof,isBDof))
bdIdx = np.zeros((A.shape[0],),np.int)
bdIdx[isBDof] = 1
Tbd = spdiags(bdIdx,0,A.shape[0],A.shape[0])
T = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
A = T@A + Tbd
#print(A.toarray())
#print(np.abs(A.toarray()).sum())

#组装第二个方程的左端矩阵
B1 = pspace.stiff_matrix()
isBDof = pspace.boundary_dof(threshold=is_p_boundary)
bdIdx = np.zeros((B1.shape[0],),np.int)
bdIdx[isBDof] = 1
Tbd = spdiags(bdIdx,0,B1.shape[0],B1.shape[0])
T = spdiags(1-bdIdx,0,B1.shape[0],B1.shape[0])
B =  T@B1 + Tbd

#组装第三个方程的左端矩阵
C = uspace.mass_matrix()
C = csr_matrix(scipy.linalg.block_diag(C.toarray(),C.toarray()))


ctx = DMumpsContext()
ctx.set_silent()
errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(nt): 
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    #组装第一个方程的右端向量
    fb1 = H@u0.flatten(order='F')

    fuu = u0(bcs)
    fgu = u0.grad_value(bcs)
    fbb2 = np.einsum('i,ijn,ijk,ijmk,j -> jnm',ws,uphi,fuu,fgu,cellmeasure)
    fb2 = np.zeros((ugdof,2))
    np.add.at(fb2,(ucell2dof,np.s_[:]),fbb2)
    fb2 = fb2.flatten(order='F') 

    fb3 = E@u0.flatten(order='F')
    
    fp = p0(bcs)
    fbb4 = np.einsum('i,ij,ijk,j->jk',ws,fp,ugphi[...,0],cellmeasure) 
    fbb5 = np.einsum('i,ij,ijk,j->jk',ws,fp,ugphi[...,1],cellmeasure) 
    fb4 = np.zeros((ugdof))
    fb5 = np.zeros((ugdof))
    np.add.at(fb4,ucell2dof,fbb4)
    np.add.at(fb5,ucell2dof,fbb5)
    fb4 = np.hstack((fb4,fb5))
    
    ep = p0(epbcs)[...,index]
    value = np.einsum('ij,jk->ijk',ep,n)
    ephi = uspace.face_basis(epbcs)
    evalue = np.einsum('i,ijk,ijm,j->jkm',epws,ephi,value,emeasure)
    fb5 = np.zeros((ugdof,2))
    np.add.at(fb5,(face2dof,np.s_[:]),evalue)
    fb5 = fb5.flatten(order='F')
    
    fb6 = D@u0.flatten(order='F') 

    b1 = fb1 - fb2 -fb3 + fb4 - fb5 + fb6
    
    fisbdf = uspace.boundary_dof(threshold = is_wall_boundary)
    ipoint = uspace.interpolation_points()
    ue2 = usolution(ipoint)
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
    sisbdf = pspace.boundary_dof(threshold = is_in_flow_boundary)
    b2[sisbdf] = 8
    sisbdf = pspace.boundary_dof(threshold = is_out_flow_boundary)
    b2[sisbdf] = 0
    
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
   
    ps = smesh.bc_to_point(bcs)
    val = usolution(ps)
    val = (val - u1(bcs))**2
    val = np.sum(val,axis=2)
    l2 = np.einsum('i,ij,j->',ws,val,cellmeasure)
    l2 = np.sqrt(l2)
    print(l2)

    uc1 = usolution(smesh.node)
    NN = smesh.number_of_nodes()
    uc2 = u1[:NN]
    up1 = psolution(smesh.node)
    up2 = p1[:NN]
    errorMatrix[0,i] = uspace.integralalg.L2_error(usolution,u1)
    errorMatrix[1,i] = pspace.integralalg.error(psolution,p1)
    errorMatrix[2,i] = np.abs(uc1-uc2).max()
    errorMatrix[3,i] = np.abs(up1-up2).max()
    u0[:] = u1 
    p0[:] = p1
    # 时间步进一层 
    tmesh.advance()

# 画图
ctx.destroy()
print("uL2:",errorMatrix[0,-1])
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
