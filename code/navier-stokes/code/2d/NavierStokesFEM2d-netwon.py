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
parser = argparse.ArgumentParser(description=
        """
        有限元方法求解NS方程
        """)

parser.add_argument('--udegree',
        default=2, type=int,
        help='运动有限元空间的次数, 默认为 1 次.')

parser.add_argument('--pdegree',
        default=1, type=int,
        help='压力有限元空间的次数, 默认为 1 次.')

parser.add_argument('--ns',
        default=16, type=int,
        help='空间各个方向剖分段数， 默认剖分 100 段.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--T',
        default=1, type=float,
        help='演化终止时间, 默认为 1')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')

args = parser.parse_args()
udegree = args.udegree
pdegree = args.pdegree
ns = args.ns
nt = args.nt
T = args.T
output = args.output


rho = 1
mu=1

udim = 2

pde = PDE()
smesh = MF.boxmesh2d(pde.domain(), nx=ns, ny=ns, meshtype='tri')
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

A = uspace.stiff_matrix()
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
    #M = bmat([[E+A+D,None,-B1],[None,E+A+D,-B2],[-B1.T,-B2.T,None]],format='csr')
    
    #右端项
    
    fuu = u0(bcs)
    fgu = u0.grad_value(bcs)
    fbb2 = np.einsum('i,ijk,ijn,ijmn,j -> jkm',ws,uphi,fuu,fgu,cellmeasure)
    fb1 = np.zeros((ugdof,2))
    np.add.at(fb1,(ucell2dof,np.s_[:]),fbb2)
    
    F = uspace.source_vector(pde.source,dim=udim) + E@u0 + fb1
    FF = np.r_['0', F.T.flat, np.zeros(pgdof)]

    u_isBdDof = uspace.is_boundary_dof(threshold=pde.is_wall_boundary)
    #p_isBdDof = np.zeros(pgdof,dtype=np.bool)
    p_isBdDof = pspace.is_boundary_dof(threshold=pde.is_p_boundary)
    
    x = np.zeros(gdof,np.float)
    ipoint = uspace.interpolation_points()
    uso = pde.u_dirichlet(ipoint)
    x[0:ugdof][u_isBdDof] = uso[:,0][u_isBdDof]
    x[ugdof:2*ugdof][u_isBdDof] = uso[u_isBdDof][:,1]
    ipoint = pspace.interpolation_points()
    pso = pde.p_dirichlet(ipoint)
    x[-pgdof:][p_isBdDof] = pso[p_isBdDof]

    isBdDof = np.hstack([u_isBdDof, u_isBdDof, p_isBdDof])
    
    FF -= M@x
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    M = T@M@T + Tbd
    FF[isBdDof] = x[isBdDof]

    
    #x[:] = spsolve(M, FF)
    ctx.set_centralized_sparse(M)
    x = FF.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    u1[:, 0] = x[:ugdof]
    u1[:, 1] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
    
    uc1 = pde.velocity(smesh.node)
    NN = smesh.number_of_nodes()
    uc2 = u1[:NN]
    up1 = pde.pressure(smesh.node)
    up2 = p1[:NN]
    
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    smesh.nodedata['velocity'] = u1
    smesh.to_vtk(fname=fname)
   
    errorMatrix[0,i] = uspace.integralalg.L2_error(pde.velocity,u1)
    errorMatrix[1,i] = pspace.integralalg.error(pde.pressure,p1)
    errorMatrix[2,i] = np.abs(uc1-uc2).max()
    errorMatrix[3,i] = np.abs(up1-up2).max()

    u0[:] = u1 
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


