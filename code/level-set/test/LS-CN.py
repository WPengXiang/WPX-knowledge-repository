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



M = space.mass_matrix()
C = space.convection_matrix(c = u).T 
A = M + dt/2*C
S = space.stiff_matrix()
fname = './test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['界面'] = phi0 
mesh.nodedata['速度'] = u 
mesh.to_vtk(fname=fname)

ctx = DMumpsContext()
ctx.set_silent()

phi1= space.function()
for i in range(0,nt):
        
    t1 = timeline.next_time_level()
    print("t1=", t1)

    ctx.set_centralized_sparse(A)
    b = M@phi0 - dt/2*(C@phi0)
    ctx.set_rhs(b)
    ctx.run(job=6)
    phi0[:] = b
    """
    ##重置
    cenbcs = np.array([1/3,1/3,1/3])
    value = np.mean(np.sqrt(np.sum(phi0.grad_value(cenbcs)**2,axis=-1)))
    print("重置前\n",value) 
    phi1[:] =phi0
    phi2= space.function()
    phi2[:] = phi0
    error = 10 
    if value>1.1:
        while True: 
            dt2 = 0.001
            alpha = 0.005
            
            @barycentric
            def signp(bcs):
                eps = 0.01
                gg = np.sqrt(np.sum(phi0.grad_value(bcs)**2,axis=-1))
                value = phi0(bcs)/np.sqrt(phi0(bcs)**2+eps**2*gg**2)
                return value
            @barycentric
            def f(bcs):
                grad = phi1.grad_value(bcs)
                val = 1 - np.sqrt(grad[...,0]**2+grad[...,1]**2)
                val *=  np.sign(phi0(bcs))
                return val
            

            b1 = space.source_vector(f)
            b1 *= dt2
            b1 += M@phi1
            b1 -= dt2*alpha*(S@phi1)

            ctx.set_centralized_sparse(M)
            ctx.set_rhs(b1)
            ctx.run(job=6)
            phi1[:] = b1

            errold = error
            error = space.integralalg.error(phi2, phi1)/dt2
            value = np.mean(np.sqrt(np.sum(phi1.grad_value(cenbcs)**2,axis=-1)))
            print("重置后\n",value) 
            print("误差\n",error)
            if  error<0.4:
                phi0[:] = phi1
                break       
            else:
                phi2[:] = phi1
    """ 
    fname = './test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.nodedata['界面'] = phi0 
    mesh.to_vtk(fname=fname)

    # 时间步进一层 
    timeline.advance()
ctx.destroy()







