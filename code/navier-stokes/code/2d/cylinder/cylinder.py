#!/usr/bin/env python3
# 

import argparse

from fealpy.decorator import cartesian,barycentric

from fealpy.boundarycondition import NeumannBC 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.decorator import cartesian

from fealpy.mesh import MeshFactory as MF

from fealpy.timeintegratoralg import UniformTimeLine

from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate

from scipy.sparse.linalg import spsolve

from fealpy.geometry import dcircle,drectangle,ddiff,dmin
from fealpy.geometry import DistDomain2d
from fealpy.mesh import DistMesh2d
import matplotlib.pyplot as plt



## 参数解析
parser = argparse.ArgumentParser(description=
        """
        单纯形网格（三角形、四面体）网格上任意次有限元方法求解热传导方程
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--dim',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--ns',
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
ns = args.ns
nt = args.nt
output = args.output

##网格
h0 = 0.05
fd = lambda p: ddiff(drectangle(p,[0, 2.20, 0, 0.41]),dcircle(p,[0.2, 0.2], 0.05))
    
def fh(p):
    h = np.sqrt((p[:,0]-0.2)**2+(p[:,1]-0.2)**2)-0.05
    h[h<0.05] = 0.02
    return h

bbox = [0, 3, 0, 0.5]
pfix = np.array([(0,0),(2.20,0),(2.20,0.41),(0,0.41)], dtype=np.float64)
domain = DistDomain2d(fd,fh,bbox,pfix)
distmesh = DistMesh2d(domain, h0)
distmesh.run()
smesh = distmesh.mesh

fig = plt.figure()
axes = fig.gca()
smesh.add_plot(axes)
plt.show()

