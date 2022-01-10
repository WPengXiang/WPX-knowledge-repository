
import numpy as np

import matplotlib.pyplot as plt
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack
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
u1 = uspace.function(dim=2)

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
#print(H)

##E矩阵
E0 = np.einsum('i,ijk,ijm,j -> jkm',ws,gphi[...,0],gphi[...,0],cellmeasure)
E1 = np.einsum('i,ijk,ijm,j -> jkm',ws,gphi[...,1],gphi[...,1],cellmeasure)
E2 = np.einsum('i,ijk,ijm,j -> jkm',ws,gphi[...,1],gphi[...,0],cellmeasure)
E00 = csr_matrix(((E0+1/2*E1).flat,(I.flat,J.flat)),shape=(gdof,gdof))
E11 = csr_matrix(((E1+1/2*E0).flat,(I.flat,J.flat)),shape=(gdof,gdof))
E10 = csr_matrix(((1/2*E2).flat,(I.flat,J.flat)),shape=(gdof,gdof))
E = vstack([hstack([E00,E10]),hstack([E10,E11])])

A = (rho/dt)*H+0.5*mu*E



##G矩阵
G0 =  np.einsum('i,ijk,ijm,ijk,j -> jkm',ws,phi,phi,gphi[...,0],cellmeasure)
G1 =  np.einsum('i,ijk,ijm,ijk,j -> jkm',ws,phi,phi,gphi[...,1],cellmeasure)
G00 = csr_matrix((G0.flat,(I.flat,J.flat)),shape=(gdof,gdof))
G11 = csr_matrix((G1.flat,(I.flat,J.flat)),shape=(gdof,gdof))
G = csr_matrix(scipy.linalg.block_diag(G00.toarray(),G11.toarray()))

G0 =  np.einsum('i,ijk,ijm,ijk,j -> jkm',ws,phi,phi,gphi[...,0],cellmeasure)


u = u0.flatten(order='F')
uxux = u*u
uxuy = u0[:,0]*u0[:,1]
uxuy = np.hstack([uxuy,uxuy])
F = -rho*G@(uxux+uxuy)+ (rho/dt*H-mu*E)@u

uspace.set_dirichlet_bc(walldirichlet,u0,threshold = is_wall_boundary)
bc = DirichletBC(uspace,walldirichlet,threshold=is_wall_boundary)

A,F = bc.apply(A,F,u1)
print(A.toarray())


fig = plt.figure()
axes = fig.gca()
smesh.add_plot(axes)
smesh.find_node(axes,node=ipoint,showindex=True)
smesh.find_cell(axes,showindex=True)
plt.show()

for i in range(0, nt): 
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    u0[:] = 0.0
    u1[:] = uh

    时间步进一层 
    tmesh.advance()
'''
