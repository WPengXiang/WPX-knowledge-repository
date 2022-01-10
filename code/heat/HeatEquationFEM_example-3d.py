#!/usr/bin/env python3
# 
import argparse
from fealpy.decorator import cartesian,barycentric
from fealpy.boundarycondition import NeumannBC 

import numpy as np
import matplotlib.pyplot as plt

# 网格工厂：生成常用的简单区域上的网格
from fealpy.mesh import MeshFactory as MF

# 均匀剖分的时间离散
from fealpy.timeintegratoralg import UniformTimeLine

# Lagrange 有限元空间
from fealpy.functionspace import LagrangeFiniteElementSpace

# Dirichlet 边界条件
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

# solver
from scipy.sparse.linalg import spsolve


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法求解热传导方程
        """)

parser.add_argument('--T',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--dim',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--NS',
        default=10, type=int,
        help='空间各个方向剖分段数， 默认剖分 10 段.')

parser.add_argument('--nt',
        default=100, type=int,
        help='时间剖分段数，默认剖分 100 段.')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')

args = parser.parse_args()
degree = args.degree
dim = args.dim
NS = args.NS
nt = args.nt
T = args.T
output = args.output
#网格
domain = [0,0.15,0,0.05,0,0.05]
smesh = MF.boxmesh3d(box=domain,nx=3*NS,ny=NS,nz=NS,meshtype='tet')


c = 200
rho = 5e3
k = 80
def init(p):
    shape = p.shape[0]
    value = np.zeros(shape,dtype=np.float64)
    value[:] = 20.0
    return value

@cartesian
def is_Neuman(p):
    eps = 1e-14
    return (np.abs(p[..., 0])<eps) | (np.abs(p[...,0]-0.15) < eps)

@cartesian
def neumann(p,n):
    shape = p.shape[0:2]
    value = np.zeros(shape, dtype=np.float64)
    value[:] = 5e5
    return value

tmesh = UniformTimeLine(0, T, nt) # 均匀时间剖分

space = LagrangeFiniteElementSpace(smesh, p=degree)

uh0 = space.interpolation(init)

# 下一层时间步的有限元解
uh1 = space.function()

A = k*space.stiff_matrix()
M = rho*c*space.mass_matrix() # 质量矩阵
dt = tmesh.current_time_step_length() # 时间步长
G = M + dt*A # 隐式迭代矩阵

# 当前时间步的有限元解
uh0 = space.interpolation(init)

# 下一层时间步的有限元解
uh1 = space.function()
fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
smesh.nodedata['temp'] = uh0 
smesh.to_vtk(fname=fname)


for i in range(0, nt): 
    
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)


    # t1 时间层的右端项
    F = M@uh0
    
    # 代数系统求解
    bc = NeumannBC(space, neumann,threshold=is_Neuman) 
    F = bc.apply(F) 
    uh1[:] = spsolve(G, F).reshape(-1)

    bc = NeumannBC(space, neumann,threshold=is_Neuman) 
    F = bc.apply(F) 

    # t1 时间层的误差
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    smesh.nodedata['temp'] = uh1 
    smesh.to_vtk(fname=fname)
    uh0[:] = uh1
    uh1[:] = 0.0

    # 时间步进一层 
    tmesh.advance()


