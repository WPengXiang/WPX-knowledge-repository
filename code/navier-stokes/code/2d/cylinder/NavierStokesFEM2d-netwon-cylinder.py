import argparse
import sys
import numpy as np
import matplotlib

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from mumps import DMumpsContext

from fealpy.mesh import MeshFactory as MF

import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.boundarycondition import DirichletBC 
from fealpy.timeintegratoralg import UniformTimeLine
## Stokes model
from navier_stokes_mold_2d import FlowPastCylinder as PDE

## error anlysis tool
from fealpy.tools import showmultirate

# 参数设置
parser = argparse.ArgumentParser(description=
        """
        有限元方法求解NS方程
        """)

parser.add_argument('--udegree',
        default=2, type=int,
        help='运动有限元空间的次数, 默认为 2 次.')

parser.add_argument('--pdegree',
        default=1, type=int,
        help='压力有限元空间的次数, 默认为 1 次.')

parser.add_argument('--nt',
        default=5000, type=int,
        help='时间剖分段数，默认剖分 5000 段.')

parser.add_argument('--T',
        default=5, type=float,
        help='演化终止时间, 默认为 5')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')

args = parser.parse_args()
udegree = args.udegree
pdegree = args.pdegree
nt = args.nt
T = args.T
output = args.output

# 网格
points = np.array([[0.0, 0.0], [2.2, 0.0], [2.2, 0.41], [0.0, 0.41]],
        dtype=np.float64)
facets = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int_)


p, f = MF.circle_interval_mesh([0.2, 0.2], 0.1, 0.01) 

points = np.append(points, p, axis=0)
facets = np.append(facets, f+4, axis=0)


fm = np.array([0, 1, 2, 3])

smesh = MF.meshpy2d(points, facets, 0.01, hole_points=[[0.2, 0.2]], facet_markers=fm, meshtype='tri')

fig = plt.figure()
axes = fig.gca()
smesh.add_plot(axes)
plt.show()


rho = 1
mu=0.001

udim = 2
pde = PDE()
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt

uspace = LagrangeFiniteElementSpace(smesh,p=udegree)
pspace = LagrangeFiniteElementSpace(smesh,p=pdegree)

u0 = uspace.function(dim=udim)
u1 = uspace.function(dim=udim)

p1 = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = pgdof+2*ugdof

A = mu*uspace.stiff_matrix()
B1,B2 = uspace.div_matrix(pspace)

C = pspace.mass_matrix()
E = (1/dt)*uspace.mass_matrix()

qf = smesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = smesh.entity_measure('cell')
uphi = uspace.basis(bcs)
ugphi = uspace.grad_basis(bcs)
ucell2dof = uspace.cell_to_dof()
ugdof = uspace.number_of_global_dofs()

ctx = DMumpsContext()
ctx.set_silent()
errorMatrix = np.zeros((4,nt),dtype=np.float64)

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
smesh.nodedata['velocity'] = u0 
smesh.nodedata['pressure'] = p1
smesh.to_vtk(fname=fname)

for i in range(0,nt):

    # 下一个时间层t1
    t1 = tmesh.next_time_level()
    print("t1=",t1)

    #左端项
    
    D1,D2 = uspace.div_matrix(uspace)
    D = D1 * np.broadcast_to(u0[...,0],D1.shape)+\
        D2 * np.broadcast_to(u0[...,1],D1.shape) 
    
    a5 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,0,0],cellmeasure)
    a6 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,0,1],cellmeasure)
    a7 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,1,0],cellmeasure)
    a8 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,1,1],cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = a5.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = a5.shape)
    A40 = csr_matrix((a5.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A41 = csr_matrix((a6.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A42 = csr_matrix((a7.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A43 = csr_matrix((a8.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    
    M = bmat([[E+A+D+A40,A42,-B1],\
            [A41,E+A+D+A43,-B2],\
            [-B1.T,-B2.T,None]],format='csr')
    
    #右端项
    
    fuu = u0(bcs)
    fgu = u0.grad_value(bcs)
    fbb2 = np.einsum('i,ijk,ijn,ijmn,j -> jkm',ws,uphi,fuu,fgu,cellmeasure)
    fb1 = np.zeros((ugdof,2))
    np.add.at(fb1,(ucell2dof,np.s_[:]),fbb2)
    
    F = uspace.source_vector(pde.source,dim=udim) + E@u0 + fb1
    FF = np.r_['0', F.T.flat, np.zeros(pgdof)]
    #边界条件处理 
    
    # 边界真解
    x = np.zeros(gdof, np.float64)

    u_isbddof_u0 = uspace.is_boundary_dof()
    u_isbddof_in = uspace.is_boundary_dof(threshold = pde.is_inflow_boundary)
    u_isbddof_out = uspace.is_boundary_dof(threshold = pde.is_outflow_boundary)
    u_isbddof_circle = uspace.is_boundary_dof(threshold = pde.is_circle_boundary)

    u_isbddof_u0[u_isbddof_in] = False 
    u_isbddof_u0[u_isbddof_out] = False 
    x[0:ugdof][u_isbddof_u0] = 0
    x[ugdof:2*ugdof][u_isbddof_u0] = 0

    u_isbddof = u_isbddof_u0
    u_isbddof[u_isbddof_in] = True
    ipoint = uspace.interpolation_points()[u_isbddof_in]
    #ipoint = uspace.interpolation_points()[u_isbddof_u0[u_isbddof_in]]
    #np.set_printoptions(threshold=10000)
    #print(ipoint)
    uinfow = pde.u_inflow_dirichlet(ipoint)
    x[0:ugdof][u_isbddof_in] = uinfow[:,0]
    x[ugdof:2*ugdof][u_isbddof_in] = uinfow[:,1]

    p_isBdDof_p0 = pspace.is_boundary_dof(threshold = pde.is_outflow_boundary) 
    x[2*ugdof:][p_isBdDof_p0] = 0 
    isBdDof = np.hstack([u_isbddof, u_isbddof, p_isBdDof_p0])

    FF -= M@x
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    M = T@M@T + Tbd
    FF[isBdDof] = x[isBdDof]
    
    ctx.set_centralized_sparse(M)
    xx = FF.copy()
    ctx.set_rhs(xx)
    ctx.run(job=6)
    u1[:, 0] = xx[:ugdof]
    u1[:, 1] = xx[ugdof:2*ugdof]
    p1[:] = xx[2*ugdof:]
     
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    smesh.nodedata['velocity'] = u1
    smesh.nodedata['pressure'] = p1
    smesh.to_vtk(fname=fname)
   
    u0[:] = u1 
    tmesh.advance()
ctx.destroy()
'''
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

'''
