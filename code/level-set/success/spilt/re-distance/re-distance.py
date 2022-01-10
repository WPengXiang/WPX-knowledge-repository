#!/usr/bin/python3
'''!    	
@Author: wpx
@File Name: level.py
@Mail: wpx15673207315@gmail.com 
@Created Time: 2021年11月19日 星期五 11时42分52秒
@bref:
@ref:
    1.sign函数重写
'''  

import argparse 
import numpy as np

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric

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

parser.add_argument('--alpha',
        default=0.0001, type=float,
        help='演化终止时间, 默认为 1')


args = parser.parse_args()
dim = args.dim
degree = args.degree
dt = args.dt
ns = args.ns
alpha = args.alpha

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


phi1 = space.function()
phi2 = space.function()

phi1[:] = phi0

S = space.stiff_matrix()
M = space.mass_matrix()
ctx = DMumpsContext()
ctx.set_silent()
ctx.set_centralized_sparse(M)
while True:

    @barycentric
    def f(bcs):
        grad = phi1.grad_value(bcs)
        val = 1 - np.sqrt(np.sum(grad**2, -1))
        val *=  np.sign(phi0(bcs))
        return val

    b = space.source_vector(f)
    b *= dt
    b += M@phi1
    b -= dt*alpha*(S@phi1)
    
    ctx.set_rhs(b)
    ctx.run(job=6)
    phi1[:] = b

    cenbcs = np.array([1/3,1/3,1/3])
    value = np.mean(np.sqrt(np.sum(phi2.grad_value(cenbcs)**2,axis=-1)))
    error = space.integralalg.error(phi2, phi1)
    print("界面梯度值",value)
    print("误差",error)
    if error <= 0.0001:
        break       
    else:
        phi2[:] = phi1









