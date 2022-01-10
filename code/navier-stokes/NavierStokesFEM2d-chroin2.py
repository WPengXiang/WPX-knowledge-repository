import numpy as np

import scipy 

import matplotlib.pyplot as plt
import matplotlib

from  fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric
from fealpy.boundarycondition import DirichletBC

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags

from fealpy.tools.show import showmultirate

## 参数解析
degree = 2
dim = 2
ns = 16
nt = 100

eps = 1e-14
T = 10
rho = 1
mu = 1
inp = 8.0
outp = 0.0

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
    pp = np.zeros(p.shape)
    pp = 8*(1-x)
    return pp

domain = [0, 1, 0, 1]

smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
tmesh = UniformTimeLine(0, T, nt) # 均匀时间剖分

uspace = LagrangeFiniteElementSpace(smesh, p=degree)
pspace = LagrangeFiniteElementSpace(smesh, p=degree)

u0 = uspace.function(dim=2)
us = uspace.function(dim=2)
u1 = uspace.function(dim=2)

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
## 压力空间
pphi = pspace.basis(bcs)
pgdof = pspace.number_of_global_dofs()
pcell2dof = pspace.cell_to_dof()

#组装第一个方程的左端矩阵
HH = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,uphi,cellmeasure)
I = np.broadcast_to(ucell2dof[:,:,None],shape = HH.shape)
J = np.broadcast_to(ucell2dof[:,None,:],shape = HH.shape)
H = csr_matrix((HH.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
H = csr_matrix(scipy.linalg.block_diag(H.toarray(),H.toarray()))

E0 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],ugphi[...,0],cellmeasure)
E1 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],ugphi[...,1],cellmeasure)
E00 = csr_matrix(((E0+E1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E11 = csr_matrix(((E0+E1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
E = csr_matrix(scipy.linalg.block_diag(E00.toarray(),E11.toarray()))

A = (rho/dt)*H+mu*E
##边界处理
isBDof = uspace.boundary_dof()
isBDof = np.hstack((isBDof,isBDof))
bdIdx = np.zeros((A.shape[0],),np.int)
bdIdx[isBDof] = 1
Tbd = spdiags(bdIdx,0,A.shape[0],A.shape[0])
T = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
A = T@A + Tbd

#组装第二个方程的左端矩阵
B = pspace.stiff_matrix()

#组装第三个方程的左端矩阵
C = uspace.mass_matrix()
C = csr_matrix(scipy.linalg.block_diag(C.toarray(),C.toarray()))
isBDof = uspace.boundary_dof()
isBDof = np.hstack((isBDof,isBDof))
bdIdx = np.zeros((C.shape[0],),np.int)
bdIdx[isBDof] = 1
Tbd = spdiags(bdIdx,0,C.shape[0],C.shape[0])
T = spdiags(1-bdIdx,0,C.shape[0],C.shape[0])
C = T@C + Tbd

errorMatrix = np.zeros((2,nt),dtype=np.float64)

for i in range(nt): 
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    #组装第一个方程的右端向量
    fuu = u0(bcs)
    fbb1 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,fuu,cellmeasure)
    fb1 = np.zeros((ugdof,2))
    np.add.at(fb1,(ucell2dof,np.s_[:]),fbb1)
    fb1 = fb1/dt

    fgu = u0.grad_value(bcs)
    fbb2 = np.einsum('i,ijn,ijk,ijmk,j -> jnm',ws,uphi,fuu,fgu,cellmeasure)
    fb2 = np.zeros((ugdof,2))
    np.add.at(fb2,(ucell2dof,np.s_[:]),fbb2)
    b1 = (fb1 - fb2).flatten(order='F')

    fisbdf = uspace.boundary_dof()
    ipoint = uspace.interpolation_points()
    ue2 = usolution(ipoint)
    b1[0:len(fisbdf)][fisbdf] = ue2[fisbdf][:,0]
    b1[len(fisbdf):][fisbdf] = ue2[fisbdf][:,1]

    fus= spsolve(A,b1)
    us[:,0] = fus[0:ugdof]
    us[:,1] = fus[ugdof:]

    #组装第二个方程的右端向量
    b2 = pspace.source_vector(us.div_value)
    b2 = -(1/dt)*b2

    pp = spsolve(B,b2)
    p1[:] = pp[:]

    #组装第三个方程的右端向量
    tuu = us(bcs)
    tbb1 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,tuu,cellmeasure)
    tb1 = np.zeros((ugdof,2))
    np.add.at(tb1,(ucell2dof,np.s_[:]),tbb1)
    
    gp = p1.grad_value(bcs)
    tbb2 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,gp,cellmeasure)
    tb2 = np.zeros((ugdof,2))
    np.add.at(tb2,(ucell2dof,np.s_[:]),tbb2)
    b3 = tb1-dt*tb2
    b3 = b3.flatten(order='F')
    b3[0:len(fisbdf)][fisbdf] = ue2[fisbdf][:,0]
    b3[len(fisbdf):][fisbdf] = ue2[fisbdf][:,1]
    
    tu1 = spsolve(C,b3)
    u1[:,0] = tu1[0:ugdof]
    u1[:,1] = tu1[ugdof:]
     
    co1 = usolution(smesh.node)
    NN = smesh.number_of_nodes()
    co2 = u1[:NN]
    errorMatrix[0,i] = (co1-co2).max()
    #errorMatrix[0,i] = uspace.integralalg.error(usolution,u1)
    errorMatrix[1,i] = pspace.integralalg.error(psolution,p1)

    u0[:] = u1 
    # 时间步进一层 
    tmesh.advance()
print(errorMatrix[1,:])

# 画图
fig1 = plt.figure()
node = smesh.node
x = tuple(node[:,0])
y = tuple(node[:,1])
NN = smesh.number_of_nodes()
u = u1[:NN]
ux = tuple(u[:,0])
uy = tuple(u[:,1])
uux = usolution(node)[:,0]
uuy = usolution(node)[:,1]

#o = ux
o = np.abs(uux-ux)
norm = matplotlib.colors.Normalize()
cm = matplotlib.cm.copper
sm = matplotlib.cm.ScalarMappable(cmap=cm,norm=norm)
sm.set_array([])
#plt.quiver(x,y,ux,uy,color=cm(norm(o)))
plt.quiver(x,y,np.abs(uux-ux),np.abs(uuy-uy),color=cm(norm(o)))
plt.colorbar(sm)
plt.show()
