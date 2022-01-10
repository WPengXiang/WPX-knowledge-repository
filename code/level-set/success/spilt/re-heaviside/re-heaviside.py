#!/usr/bin/python3
'''!    	
@Author: wpx
@File Name: level.py
@Mail: wpx15673207315@gmail.com 
@Created Time: 2021年11月19日 星期五 11时42分52秒
@ref:
@bref:
'''  

import argparse 
import numpy as np

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric

from scipy.sparse import csr_matrix
from mumps import DMumpsContext

## 参数解析
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

parser.add_argument('--dt',
        default=0.01, type=int,
        help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--epsilon',
        default=0.0001, type=float,
        help='演化终止时间, 默认为 1')


args = parser.parse_args()
dim = args.dim
degree = args.degree
dt = args.dt
ns = args.ns
epsilon = args.epsilon

@cartesian
def test2d(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    return val

@cartesian
def test3d(p):
    x = p[...,0]
    y = p[...,1]
    z = p[...,2]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2+(z-0.5)**2)-0.15
    return val


if dim == 2:
    domain = [0, 1, 0, 1]
    mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    phi0 = space.interpolation(test2d)
else:
    domain = [0, 1, 0, 1, 0, 1]
    mesh = MF.boxmesh3d(domain, nx=ns, ny=ns, nz=ns, meshtype='tet')
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    phi0 = space.interpolation(test3d)

qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')

phi = space.basis(bcs)
gphi = space.grad_basis(bcs)
gdof = space.number_of_global_dofs()
cell2dof = space.cell_to_dof()

phi1 = phi0
phi2 = space.function()

## 组装A
grad = phi0.grad_value(bcs)
@barycentric
def n(bcs):
    n = np.sqrt(np.sum(grad**2,-1))[...,np.newaxis]
    tag = grad==0
    n = grad/n
    n[tag] = 0
    return n

M = space.mass_matrix()
C = space.convection_matrix(c=n)

def nn(bcs):
    val = np.einsum('ijm,ijkm->ijk',n(bcs),space.grad_basis(bcs))
    return val

A2 = np.einsum('i,ijk,ijm,j->jkm',ws,nn(bcs),nn(bcs),cellmeasure) 
I = np.broadcast_to(cell2dof[:,:,None],shape = A2.shape)
J = np.broadcast_to(cell2dof[:,None,:],shape = A2.shape)
A2 = csr_matrix((A2.flat,(I.flat,J.flat)),shape=(gdof,gdof))

ctx = DMumpsContext()
ctx.set_silent()

while True:             
    @barycentric
    def fun(bcs):
        val = np.einsum('ijk,ij->ijk',n(bcs),phi1(bcs))
        return val
    
    C1 = space.convection_matrix(c=fun)
    
    A = M - (dt/2)*C + (epsilon*dt/2)*A2 + dt*C1
    b = M@phi1 + (dt/2)*C*phi1 - (epsilon*dt/2)*A2@phi1
      
    ctx.set_centralized_sparse(A)
    ctx.set_rhs(b)
    ctx.run(job=6)
    phi2[:] = b
    
    error = space.integralalg.error(phi2, phi1)
    print("误差",error)
    
    if error <= 0.0001:
        break       
    else:
        phi1[:] = phi2







