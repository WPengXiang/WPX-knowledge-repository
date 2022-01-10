import argparse
import sys
import numpy as np
import matplotlib

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from mumps import DMumpsContext

from fealpy.decorator import cartesian,barycentric
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.boundarycondition import DirichletBC 
from fealpy.timeintegratoralg import UniformTimeLine
## Stokes model
from navier_stokes_mold_3d import Poisuille as PDE

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

udim = 3

pde = PDE()
smesh = MF.boxmesh3d(pde.domain(), nx=ns, ny=ns,nz=ns, meshtype='tet')
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
gdof = pgdof+3*ugdof

A = uspace.stiff_matrix()
B1,B2,B3 = uspace.div_matrix(pspace)

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
    a00 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,0,0],cellmeasure)
    a01 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,0,1],cellmeasure)
    a02 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,0,2],cellmeasure)
    a10 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,1,0],cellmeasure)
    a11 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,1,1],cellmeasure)
    a12 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,1,2],cellmeasure)
    a20 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,2,0],cellmeasure)
    a21 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,2,1],cellmeasure)
    a22 = np.einsum('i,ijk,ijm,ij,j->jkm',ws,uphi,\
            uphi,u0.grad_value(bcs)[...,2,2],cellmeasure)

    I = np.broadcast_to(ucell2dof[:,:,None],shape = a00.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = a00.shape)

    A00 = csr_matrix((a00.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A01 = csr_matrix((a01.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A02 = csr_matrix((a02.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A10 = csr_matrix((a10.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A11 = csr_matrix((a11.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A12 = csr_matrix((a12.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A20 = csr_matrix((a20.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A21 = csr_matrix((a21.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    A22 = csr_matrix((a22.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
    
    D1,D2,D3 = uspace.div_matrix(uspace)
    D = D1 * np.broadcast_to(u0[...,0],D1.shape)+\
        D2 * np.broadcast_to(u0[...,1],D1.shape)+\
        D3 * np.broadcast_to(u0[...,2],D1.shape)
        
    M = bmat([[E+A+D+A00,A01,A02,-B1],\
              [A10,E+A+D+A11,A12,-B2],\
              [A20,A21,E+A+D+A22,-B3],\
              [-B1.T,-B2.T,-B3.T,None]],format='csr')
   
    #右端项
    @barycentric
    def f(bcs):
        value = np.einsum('ijn,ijmn->ijm',u0(bcs),u0.grad_value(bcs))
        return value
    
    fb1 = uspace.source_vector(f,dim=3)
    F = uspace.source_vector(pde.source,dim=udim) + E@u0 + fb1
    FF = np.r_['0', F.T.flat, np.zeros(pgdof)]

    u_isBdDof = uspace.is_boundary_dof()
    #p_isBdDof = np.zeros(pgdof,dtype=np.bool)
    p_isBdDof = pspace.is_boundary_dof(threshold=pde.is_p_boundary)
    
    x = np.zeros(gdof,np.float)
    ipoint = uspace.interpolation_points()
    uso = pde.u_dirichlet(ipoint)
    x[0:ugdof][u_isBdDof] = uso[u_isBdDof][:,0]
    x[ugdof:2*ugdof][u_isBdDof] = uso[u_isBdDof][:,1]
    x[2*ugdof:3*ugdof][u_isBdDof] = uso[u_isBdDof][:,2]
    ipoint = pspace.interpolation_points()
    pso = pde.p_dirichlet(ipoint)
    x[-pgdof:][p_isBdDof] = pso[p_isBdDof]

    isBdDof = np.hstack([u_isBdDof, u_isBdDof, u_isBdDof, p_isBdDof])
    
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
    u1[:, 1] = x[2*ugdof:3*ugdof]
    p1[:] = x[3*ugdof:]
    
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    smesh.nodedata['velocity'] = u1 
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

    u0[:] = u1 

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
