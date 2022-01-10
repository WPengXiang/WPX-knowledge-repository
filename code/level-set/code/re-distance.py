#/usr/bin/python3
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
import matplotlib.pyplot as plt

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric

from mumps import DMumpsContext

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        有限元方法重置水平集函数
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
        default=0.0001, type=float,
        help='时间步长.')

parser.add_argument('--alpha',
        default=0.00625, type=float,
        help='人工粘性项系数')

parser.add_argument('--steps',
        default=10, type=int,
        help='水平集重新初始化的次数，默认 10 次')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')


args = parser.parse_args()
dim = args.dim
degree = args.degree
dt = args.dt
ns = args.ns
steps = args.steps
alpha = 0.625/ns 
epsilon = 1.0/ns
output = args.output

@cartesian
def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = (x-0.5)**2+(y-0.75)**2 - 0.15**2
    return val

@cartesian
def sphere(p):
    x = p[...,0]
    y = p[...,1]
    z = p[...,2]
    val = (x-0.5)**2 + (y-0.75)**2 +(z-0.5)**2 - 0.15**2
    return val


if dim == 2:
    domain = [0, 1, 0, 1]
    mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    phi0 = space.interpolation(circle)
else:
    domain = [0, 1, 0, 1, 0, 1]
    mesh = MF.boxmesh3d(domain, nx=ns, ny=ns, nz=ns, meshtype='tet')
    space = LagrangeFiniteElementSpace(mesh, p=degree)
    phi0 = space.interpolation(sphere)


phi1 = space.function()
phi2 = space.function()
phi1[:] = phi0

S = space.stiff_matrix()
M = space.mass_matrix()


bc = np.array([1/3,1/3,1/3])

val0 = np.sqrt(np.sum(phi0.grad_value(bc)**2, axis=-1))
val1 = np.sqrt(np.sum(phi1.grad_value(bc)**2, axis=-1))

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['phi0'] = phi0 
mesh.celldata['val0'] = val0
mesh.nodedata['phi1'] = phi1 
mesh.celldata['val1'] = val1
mesh.to_vtk(fname=fname)

ctx = DMumpsContext()
ctx.set_silent()
ctx.set_centralized_sparse(M)

@barycentric
def signp(bcs):
    val0 = phi0(bcs)
    grad = phi1.grad_value(bcs)
    val1 = np.sqrt(np.sum(grad**2, -1))

    val = 1 - val1 
    val *= val0
    val /= np.sqrt(val0**2 + epsilon**2*val1**2)
    return val

aerror = [0]
area = space.function()

cont = 0
E0 = 1e10
for i in range(steps):
    print("i = ", i)

    b = space.source_vector(signp)
    b *= dt
    b += M@phi1
    b -= dt*alpha*(S@phi1)
    
    #计算面积
    area[phi0 > 0] = 0
    area[phi0 <=0] = 1
    aerror.append(abs(space.integralalg.integral(area) - (np.pi)*0.15**dim))

    ctx.set_rhs(b)
    ctx.run(job=6)
    phi2[:] = b

    E = space.integralalg.error(phi2, phi1)
    print("相邻两次迭代误差:", E)

    if E0 < E:
        fail = 1
        print("求解发散!", cont)
        break
    
    cont += 1
    E0 = E

    phi1[:] = phi2

    val1 = np.sqrt(np.sum(phi1.grad_value(bc)**2, axis=-1))

    fname = output + 'test_'+ str(i).zfill(10) + '.vtu'
    mesh.nodedata['phi0'] = phi0 
    mesh.nodedata['phi1'] = phi1 
    mesh.celldata['val0'] = val0
    mesh.celldata['val1'] = val1

    mesh.to_vtk(fname=fname)

ctx.destroy()

plt.figure()
plt.plot(range(len(aerror)),  aerror,'--', color='g', label='Area error')
plt.legend(loc='upper right')
plt.savefig(fname = output+'area'+'.png')
plt.show()








