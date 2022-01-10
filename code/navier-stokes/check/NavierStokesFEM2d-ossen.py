import numpy as np
from mumps import DMumpsContext
from scipy.linalg import solve
from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as plt
import matplotlib

from  fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric
from fealpy.boundarycondition import DirichletBC
from fealpy.quadrature import GaussLegendreQuadrature
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from fealpy.tools.show import showmultirate

## 参数解析
udegree = 2
pdegree = 1
dim = 1
ns = 8
nt = 30

eps = 1e-12
T = 10
rho = 1
mu = 1
inp = 8.0
outp = 0.0


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
Nugdof = uspace.number_of_global_dofs()
## 压力空间
pphi = pspace.basis(bcs)
pgphi = pspace.grad_basis(bcs)
pgdof = pspace.number_of_global_dofs()
pcell2dof = pspace.cell_to_dof()
Npgdof = pspace.number_of_global_dofs()
#组装第一个方程的左端矩阵
E = (1/dt)*uspace.mass_matrix()

A = uspace.stiff_matrix()

B1 = np.einsum('i,ijm,ijk,j->jmk',ws,ugphi[...,0],pphi,cellmeasure)
B2 = np.einsum('i,ijm,ijk,j->jmk',ws,ugphi[...,1],pphi,cellmeasure)
I = np.broadcast_to(ucell2dof[:,:,None],shape=B1.shape)
J = np.broadcast_to(pcell2dof[:,None,:],shape=B1.shape)
B1 = csr_matrix((B1.flat,(I.flat,J.flat)),shape=(ugdof,pgdof))
B2 = csr_matrix((B2.flat,(I.flat,J.flat)),shape=(ugdof,pgdof))

C = pspace.mass_matrix()
ctx = DMumpsContext()
ctx.set_silent()
errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(1):
    t1 = tmesh.next_time_level()
    print("t=",t1)
   
    D1 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,u0(bcs)[...,0],uphi,ugphi[...,0],cellmeasure) 
    D2 = np.einsum('i,ij,ijk,ijm,j->jkm',ws,u0(bcs)[...,1],uphi,ugphi[...,1],cellmeasure) 
    I = np.broadcast_to(ucell2dof[:,:,None],shape=D1.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape=D1.shape)
    D1 = csr_matrix((D1.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    D2 = csr_matrix((D2.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    D = D1+D2
    
    M = bmat([[E+A+D,None,-B1],[None,E+A+D,-B2],[-B1.T,-B2.T,None]])
    print(M.sum())

    isuBDof = uspace.is_boundary_dof()
    ispBDof = pspace.is_boundary_dof(threshold = is_p_boundary)
    isDDof = np.hstack((isuBDof,isuBDof,ispBDof))
    bdIdx = np.zeros((M.shape[0],),np.int_)
    bdIdx[isDDof] = 1
    Tbd = spdiags(bdIdx,0,M.shape[0],M.shape[0])
    T = spdiags(1-bdIdx,0,M.shape[0],M.shape[0])
    M = T@M + Tbd
     
    b = E@u0
    b = np.hstack((b.T.flat,np.zeros(Npgdof)))

    ipoint = uspace.interpolation_points()
    ue2 = usolution(ipoint)
    b[0:len(isuBDof)][isuBDof] = ue2[isuBDof][:,0]
    b[len(isuBDof):-Npgdof][isuBDof] = ue2[isuBDof][:,1]
     
    ipoint = pspace.interpolation_points()
    pe2 = psolution(ipoint)
    b[-Npgdof:][ispBDof] = pe2[ispBDof]
    
    ctx.set_centralized_sparse(M)
    x = b.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    
    u1[:,0] = x[0:ugdof]
    u1[:,1] = x[ugdof:-Npgdof]
    p1[:] = x[-Npgdof:]
    
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

    tmesh.advance()

ctx.destroy()
print("uL2:",errorMatrix[2,-1])
print("pL2:",errorMatrix[1,-1])
print("umax:",errorMatrix[2,-1])
print("pmax:",errorMatrix[3,-1])
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

