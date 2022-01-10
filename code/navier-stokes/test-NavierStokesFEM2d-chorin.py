
import numpy as np

import matplotlib.pyplot as plt
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags
import scipy.linalg
from fealpy.decorator import cartesian , barycentric
from fealpy.boundarycondition import DirichletBC


degree = 2
dim = 2
ns = 1
nt = 50

eps = 1e-12
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

domain = [0, 1, 0, 1]

smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
tmesh = UniformTimeLine(0, T, nt) # 均匀时间剖分

uspace = LagrangeFiniteElementSpace(smesh, p=degree)
pspace = LagrangeFiniteElementSpace(smesh, p=degree)

u0 = uspace.function(dim=2)
us = uspace.function(dim=2)
u1 = uspace.function(dim=2)


ux = uspace.function()
uy = uspace.function()


p0 = pspace.function()
p1 = pspace.function()


#矩阵组装准备工作
dt = tmesh.dt
qf = smesh.integrator(4,'cell')
ipoint = uspace.interpolation_points()
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = smesh.entity_measure('cell')
phi = uspace.basis(bcs)
gphi = uspace.grad_basis(bcs)
gdof = uspace.number_of_global_dofs()
cell2dof = uspace.cell_to_dof()

##H矩阵
HH = np.einsum('i,ijk,ijm,j -> jkm',ws,phi,phi,cellmeasure)
I = np.broadcast_to(cell2dof[:,:,None],shape = HH.shape)
J = np.broadcast_to(cell2dof[:,None,:],shape = HH.shape)
H = csr_matrix((HH.flat,(I.flat,J.flat)),shape=(gdof,gdof))
H = csr_matrix(scipy.linalg.block_diag(H.toarray(),H.toarray()))
#print(H.toarray())

##E矩阵
E0 = np.einsum('i,ijk,ijm,j -> jkm',ws,gphi[...,0],gphi[...,0],cellmeasure)
E1 = np.einsum('i,ijk,ijm,j -> jkm',ws,gphi[...,1],gphi[...,1],cellmeasure)
E00 = csr_matrix(((E0+E1).flat,(I.flat,J.flat)),shape=(gdof,gdof))
E11 = csr_matrix(((E0+E1).flat,(I.flat,J.flat)),shape=(gdof,gdof))
E = csr_matrix(scipy.linalg.block_diag(E00.toarray(),E11.toarray()))
#print(E.toarray())
#print(uspace.stiff_matrix().toarray())

A = (rho/dt)*H+mu*E

#边界条件
isBDof = uspace.boundary_dof(threshold=is_wall_boundary)
isBDof = np.hstack((isBDof,isBDof))
bdIdx = np.zeros((A.shape[0],),np.int)
bdIdx[isBDof] = 1
Tbd = spdiags(bdIdx,0,A.shape[0],A.shape[0])
T = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
A = T@A + Tbd
#print(A.toarray().sum())

#第二个矩阵
B = pspace.stiff_matrix()
isBDof = pspace.boundary_dof(threshold=is_p_boundary)
bdIdx = np.zeros((B.shape[0],),np.int)
bdIdx[isBDof] = 1
Tbd = spdiags(bdIdx,0,B.shape[0],B.shape[0])
T = spdiags(1-bdIdx,0,B.shape[0],B.shape[0])
B =  T@B + Tbd
#print(B.toarray())

#第三个矩阵
C = uspace.mass_matrix()
C = csr_matrix(scipy.linalg.block_diag(C.toarray(),C.toarray()))
#print(C.toarray())

'''
##G矩阵
G0 =  np.einsum('i,ijk,ijm,ijk,j -> jkm',ws,phi,phi,gphi[...,0],cellmeasure)
G1 =  np.einsum('i,ijk,ijm,ijk,j -> jkm',ws,phi,phi,gphi[...,1],cellmeasure)
G00 = csr_matrix((G0.flat,(I.flat,J.flat)),shape=(gdof,gdof))
G11 = csr_matrix((G1.flat,(I.flat,J.flat)),shape=(gdof,gdof))
G = csr_matrix(scipy.linalg.block_diag(G00.toarray(),G11.toarray()))
u = u0.flatten(order='F')

uxux = u*u
uxuy = u0[:,0]*u0[:,1]
uxuy = np.hstack([uxuy,uxuy])
a = -rho*G@(uxux+uxuy) + (rho/dt)*H@u
a[isBDof] = 0
u = spsolve(A,a)
u.reshape((-1,2),order='F')
print(u)
'''
#第一个右端项

ux[:] = spsolve(A,b1)[0:gdof]
uxx = us(bcs)
b31 = np.einsum('i,ijk,ijm,j -> jkm',ws,phi,uxx,cellmeasure)
shape = (gdof,2)
b3 = np.zeros(shape)
np.add.at(b3,(cell2dof,np.s_[:]),b31)

pxx = p1.grad_value(bcs)
b32 = np.einsum('i,ijk,ijm,j -> jkm',ws,phi,pxx,cellmeasure)
b34 = np.zeros(shape)
np.add.at(b34,(cell2dof,np.s_[:]),b32)
b3 = b34+b3

gu = u0.grad_value(bcs)
u = u0(bcs)
b1 = np.einsum('i,ijn,ijm,ijmk,j -> jnk',ws,phi,u,gu,cellmeasure)
shape = (gdof,2)
bb1 = np.zeros(shape)
np.add.at(bb1,(cell2dof,np.s_[:]),b1)
b1 = bb1.flatten(order='F')
isbdf = uspace.boundary_dof(threshold = is_wall_boundary)
isbdf = np.hstack((isbdf,isbdf))
b1[isbdf] = 0

us[:,0] = spsolve(A,b1)[0:gdof]
us[:,1] = spsolve(A,b1)[gdof:]

#右端第二项
ux[:] = spsolve(A,b1)[0:gdof]
b2 = pspace.source_vector(us.div_value)
b2 = -b2/dt
isbdf = pspace.boundary_dof(threshold = is_in_flow_boundary)
b2[isbdf] = 8
isbdf = pspace.boundary_dof(threshold = is_out_flow_boundary)
b2[isbdf] = 0
pp = spsolve(B,b2)
p1[:] = pp[:]


#右端第三项
u = us(bcs)
b31 = np.einsum('i,ijk,ijm,j -> jkm',ws,phi,uxx,cellmeasure)
shape = (gdof,2)
b3 = np.zeros(shape)
np.add.at(b3,(cell2dof,np.s_[:]),b31)

pxx = p1.grad_value(bcs)
b32 = np.einsum('i,ijk,ijm,j -> jkm',ws,phi,pxx,cellmeasure)
b34 = np.zeros(shape)
np.add.at(b34,(cell2dof,np.s_[:]),b32)
b3 = b34+b3
 
'''
pphi = pspace.basis(bcs)
du = us.div_value(bcs)
b2 = np.einsum('i,ijk,ij,j->jk',ws,pphi,du,cellmeasure)
shape = gdof
bb2 = np.zeros(shape)
np.add.at(bb2,cell2dof,b2)
print(bb2)
'''

fig = plt.figure()
axes = fig.gca()
smesh.add_plot(axes)
smesh.find_node(axes,node=ipoint,showindex=True)
smesh.find_cell(axes,showindex=True)
plt.show()

'''
for i in range(0, nt): 
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
       
    u0[:,0] = 0.0
    u0[:,1] = 0.0
    gu0 = u0.grad_value(bcs)
    u = u0(bcs)
    va = np.einsum('i,ijn,ijm,ijmk,j -> jnk',ws,phi,u,gu,cellmeasure)
    shape = (gdof,2)
    b1 = np.zeros(shape)
    np.add.at(b1,(cell2dof,np.s_[:]),va)
    b1.reshape((-1,1),order='F')
    us[:,0] = spsolve(A,b1)[0:gdof]
    us[:,1] = spsolve(A,b1)[gdof:]
    
    u


    u1[:] = uh

    时间步进一层 
    tmesh.advance()
'''
