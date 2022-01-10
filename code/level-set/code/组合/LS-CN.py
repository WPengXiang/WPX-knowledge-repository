#!/usr/bin/python3
'''!    	
@Author: wpx
@File Name: level.py
@Mail: wpx15673207315@gmail.com 
@Created Time: 2021年11月19日 星期五 11时42分52秒
@bref:
@ref:
'''  

import argparse 
import numpy as np
import matplotlib.pyplot as plt

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric
from fealpy.mesh import LagrangeTriangleMesh
from scipy.sparse.linalg import cg

from mumps import DMumpsContext

import pickle
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

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')

parser.add_argument('--alpha',
        default=0.00625, type=float,
        help='人工粘性项系数')

parser.add_argument('--dt',
        default=0.0001, type=float,
        help='时间步长.')

parser.add_argument('--steps',
        default=10, type=int,
        help='水平集重新初始化的次数，默认 10 次')

args = parser.parse_args()

dim = args.dim
degree = args.degree
nt = args.nt
ns = args.ns
T = args.T
output = args.output
re_dt = args.dt
alpha = 0.625/ns
epsilon = 1.0/ns
steps = args.steps


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
    return val

@cartesian
def sphere(p):
    x = p[...,0]
    y = p[...,1]
    z = p[...,2]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2+(z-0.5)**2)-0.15
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

def rein(phi0,A=space.mass_matrix(),dt=re_dt):
    phi1 = space.function()
    phi2 = space.function()
    phi1[:] = phi0
    ctx.set_centralized_sparse(A)
    E0 = 1e10
    cont = 0

    @barycentric
    def signp(bcs):
        val0 = phi0(bcs)
        grad = phi1.grad_value(bcs)
        val1 = np.sqrt(np.sum(grad**2, -1))

        val = 1 - val1 
        val *= val0
        val /= np.sqrt(val0**2 + epsilon**2*val1**2)
        return val
     
    for i in range(steps):
        print("i = ", i)

        b = space.source_vector(signp)
        b *= dt
        b += M@phi1
        b -= dt*alpha*(S@phi1)
        ''' 
        #计算面积
        area[phi0 > 0] = 0
        area[phi0 <=0] = 1
        aerror.append(abs(space.integralalg.integral(area) - (np.pi)*0.15**dim))
        '''
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
        
        #val1 = np.sqrt(np.sum(phi1.grad_value(bc)**2, axis=-1))
    return phi1


S = space.stiff_matrix()
M = space.mass_matrix()
C = space.convection_matrix(c = u).T 

## LS
A = M + dt/2*C
## re
diff = []
measure = space.function()

if output != 'None':
    fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
    if degree == 1:
        mesh.nodedata['phi'] = phi0 
        mesh.nodedata['velocity'] = u 
        mesh.to_vtk(fname=fname)
    else:
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        lmesh = LagrangeTriangleMesh(node, cell, p=degree)
        lmesh.node = space.interpolation_points() 
        lmesh.ds.cell = space.cell_to_dof()
        lmesh.nodedata['phi'] = phi0 
        lmesh.nodedata['velocity'] = u 
        lmesh.to_vtk(fname=fname)

ctx = DMumpsContext()
ctx.set_silent()
ctx.set_centralized_sparse(A)
bc = np.array([1/3,1/3,1/3])
for i in range(nt):
        
    t1 = timeline.next_time_level()
    print("t1=", t1)
    
    #计算面积
    measure[phi0 > 0] = 0
    measure[phi0 <=0] = 1
    diff.append(abs(space.integralalg.integral(measure) - (np.pi)*0.15**dim))

    b = M@phi0 - dt/2*(C@phi0)
    ctx.set_rhs(b)
    ctx.run(job=6)
    phi0[:] = b
    
    gradnorm = np.mean(np.sqrt(np.sum(phi0.grad_value(bc)**2, axis=-1)))
    print(gradnorm)
    if i%5==0:
    #if np.abs(val-1)>0.01:
        phi0 = rein(phi0)
    
    if output != 'None':
        fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
        if degree == 1:
            mesh.nodedata['phi'] = phi0 
            mesh.nodedata['velocity'] = u 
            mesh.to_vtk(fname=fname)
        else:
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            lmesh = LagrangeTriangleMesh(node, cell, p=degree)
            lmesh.node = space.interpolation_points() 
            lmesh.ds.cell = space.cell_to_dof()
            lmesh.nodedata['phi'] = phi0 
            lmesh.nodedata['velocity'] = u 
            lmesh.to_vtk(fname=fname)

    # 时间步进一层 
    timeline.advance()

ctx.destroy()

pickle_file = open('diff'+str(ns)+'-'+str(degree)+'.pkl','wb')
pickle.dump(diff, pickle_file) # 写入文件
pickle_file.close()

plt.figure()
plt.plot(range(len(diff)), diff, '--', color='g', label='Measure Difference')
plt.legend(loc='upper right')
plt.savefig(fname = output+'measure'+'.png')
plt.show()


