#!/usr/bin/python3
'''！    	
@Author: wpx
@File Name: level.py
@Mail: wpx15673207315@gmail.com 
@Created Time: 2021年11月19日 星期五 11时42分52秒
@ref:
@bref:区域截断
'''  
import argparse 
import numpy as np

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric

from mumps import DMumpsContext
from scipy.sparse import csr_matrix
#参数解析
parser = argparse.ArgumentParser(description=
        """
        有限元方法求解水平集演化方程,时间离散CN格式
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--dim',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--ns',
        default=100, type=int,
        help='空间各个方向剖分段数， 默认剖分 100 段.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--T',
        default=1, type=float,
        help='演化终止时间, 默认为 1')


args = parser.parse_args()

dim = args.dim
degree = args.degree
nt = args.nt
ns = args.ns
T = args.T

epsilon = (1/ns)**0.9/2

@cartesian
def u2(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[..., 1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u


@cartesian
def u3(p):
    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[..., 1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    u[..., 2] = 0
    return u


@cartesian
def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    val = 1/(1+np.exp(-val/epsilon)) 
    return val

@cartesian
def sphere(p):
    x = p[...,0]
    y = p[...,1]
    z = p[...,2]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2+(z-0.5)**2)-0.15
    val = 1/(1+np.exp(-val/epsilon)) 
    return val


timeline = UniformTimeLine(0, T, nt)
dt = timeline.dt

if dim == 2:
    domain = [0, 1, 0, 1]
    mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    phi0 = space.interpolation(circle)
    u = space.interpolation(u2, dim=dim)
else:
    domain = [0, 1, 0, 1, 0, 1]
    mesh = MF.boxmesh3d(domain, nx=ns, ny=ns, nz=ns, meshtype='tet')
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    phi0 = space.interpolation(sphere)
    u = space.interpolation(u3, dim=dim)


qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')
gdof = space.number_of_global_dofs()
cell2dof = space.cell_to_dof()
#组装A
M = space.mass_matrix()
CU = space.convection_matrix(c=u)

norm_u = np.sum(space.integralalg.L2_norm(u,celltype=True),axis=1)
dx = min(mesh.entity_measure('edge'))
ss = dx/(2*norm_u) 

@barycentric
def s(bcs):
    val = np.einsum('ijm,j->ijm',u(bcs),ss)
    return val

CS = space.convection_matrix(c=s)

def uu(bcs):
    val = np.einsum('ijm,ijkm->ijk',u(bcs),space.grad_basis(bcs))
    return val
UU = np.einsum('i,ijk,ijm,j,j->jkm',ws,uu(bcs),uu(bcs),ss,cellmeasure) 
I = np.broadcast_to(cell2dof[:,:,None],shape = UU.shape)
J = np.broadcast_to(cell2dof[:,None,:],shape = UU.shape)
UU = csr_matrix((UU.flat,(I.flat,J.flat)),shape=(gdof,gdof))

A = M - dt/2*CU + CS + dt/2*UU

##边界处理
isBDof = space.boundary_dof()
bdIdx = np.zeros(A.shape[0],np.int)
bdIdx[isBDof] = 1
Tbd = spdiags(bdIdx,0,A.shape[0],A.shape[0])
T = spdiags(1-bdIdx,0,A.shape[0],A.shape[0])
A = T@A + Tbd


fname = './test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['界面'] = phi0
mesh.nodedata['速度'] = u 
mesh.to_vtk(fname=fname)

ctx = DMumpsContext()
ctx.set_silent()
ctx.set_centralized_sparse(A)

for i in range(0,nt):
    t1 = timeline.next_time_level()
    print("t1=",t1)

    b = M@phi0 + dt/2*CU@phi0 + CS@phi0 - dt/2*UU@phi0
    b[isBDof] = 1
    
    ctx.set_rhs(b)
    ctx.run(job=6)
    phi0[:] = b

    fname = './test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['界面'] = phi0 
    mesh.to_vtk(fname=fname)
    timeline.advance()
ctx.destroy()
