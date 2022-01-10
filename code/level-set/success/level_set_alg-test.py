#!/usr/bin/python3
'''    	
@Author: wpx
@File Name: level.py
@Author: wpx
@Mail: wpx15673207315@gmail.com 
@Created Time: 2021年11月19日 星期五 11时42分52秒
'''  

import argparse 
import numpy as np

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric

from fealpy.decorator import cartesian, barycentric
from scipy.sparse.linalg import spsolve,cg
from level_set_alg import Level_set 
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

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--T',
        default=1, type=float,
        help='演化终止时间, 默认为 1')

parser.add_argument('--outputdir',
        default='~/result/', type=str,
        help='')

args = parser.parse_args()

dim = args.dim
degree = args.degree
nt = args.nt
ns = args.ns
T = args.T

if dim == 2:
    domain = [0, 1, 0, 1]
    mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
else:
    domain = [0, 1, 0, 1, 0, 1]
    mesh = MF.boxmesh3d(domain, nx=ns, ny=ns, nz=ns, meshtype='tet')

space = LagrangeFiniteElementSpace(mesh, p=degree)
timeline = UniformTimeLine(0, T, nt)
dx = min(mesh.entity_measure('edge'))
epsilon = (dx**0.9)/2 

@cartesian
def u(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[..., 1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u


@cartesian
def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    #val = 1/(1+np.exp(-val/epsilon)) 
    return val

@cartesian
def sphere(p):
    x = p[...,0]
    y = p[...,1]
    z = p[...,2]
    val = np.sqrt((x-0.5)**2+(x-0.5)**2+(y-0.075)**2)-0.15
    return val


u = space.interpolation(u, dim=dim)

if dim == 2:
    phi = space.interpolation(circle)
else:
    phi = space.interpolation(sphere)

dt = timeline.dt

fname = './test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['phi'] = phi 
mesh.to_vtk(fname=fname)

alg = Level_set(u,dt)

A = alg.LS_CN_A()
#A = alg.CLS_SUPG_A()
for i in range(nt):
        
    t1 = timeline.next_time_level()
    print("t1=", t1)
    
    b = alg.LS_CN_b(phi)
    #b = alg.CLS_SUPG_b(phi)
    phi[:] = spsolve(A, b)
    fname = './test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['phi'] = phi 
    mesh.to_vtk(fname=fname)

    # 时间步进一层 
    timeline.advance()








