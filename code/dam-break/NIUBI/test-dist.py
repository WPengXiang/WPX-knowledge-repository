#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: test-ipcs.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2021年12月26日 星期日 12时45分16秒
	@bref 
	@ref 
'''  
import numpy as np

from fealpy.geometry import dmin
from fealpy.decorator import cartesian,barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.timeintegratoralg import UniformTimeLine

ns = 100
T = 1
nt = 100
udegree = 2
pdegree = 1
fdegree = 2
##参数设置
eps = 1e-12
domain = [0,0.584,0,0.438]
dam_domain=[0,0.146,0,0.292]
#a = 0.05715
#domain = [0,5*a,0,5*a]
#dam_domain=[0,a,0,a]
rho_water = 1000  #kg/m^3
rho_air = 1  #kg/m^3
mu_water = 0.001 #pa*s
mu_air = 0.00001 #Pa*s
g = 9.8 #m/s^2

##建立空间
mesh = MF.boxmesh2d(domain,nx=ns,ny=ns,meshtype='tri')
timeline = UniformTimeLine(0,T,nt)
uspace = LagrangeFiniteElementSpace(mesh,p=udegree)
pspace = LagrangeFiniteElementSpace(mesh,p=pdegree)
fspace = LagrangeFiniteElementSpace(mesh,p=fdegree)

@cartesian
def dist2(p):
    x = p[...,0]
    y = p[...,1]
    x0 = dam_domain[0]
    x1 = dam_domain[1]
    y0 = dam_domain[2]
    y1 = dam_domain[3]
    value = np.zeros(shape=x.shape)
    tagx =  np.logical_and(x0<=x,x<=x1)
    tagy =  np.logical_and(y0<=y,y<=y1)
    area00 = np.logical_and(tagx,tagy)
    area01 = np.logical_and(tagx,y>y1)
    area10 = np.logical_and(x>x1,tagy)
    area11 = np.logical_and(x>x1,y>y1)
    value[area00] = -np.minimum(x1-x,y1-y)[area00]
    value[area01] = (y-y1)[area01]
    value[area10] = (x-x1)[area10]
    value[area11] = np.sqrt((x-x1)**2+(y-y1)**2)[area11]
    return value

@cartesian
def dist(p):
    return -dmin(dam_domain[3]-p[:,1], dam_domain[1] - p[:,0])

cenbcs = np.array([1/3,1/3,1/3])
s0 = fspace.interpolation(dist)
s1 = fspace.interpolation(dist2)
mesh.nodedata['dist'] = s0
mesh.nodedata['dist2'] = s1
gs0 = s0.grad_value(cenbcs)
gs1 = s1.grad_value(cenbcs)
gs0 = np.sqrt(gs0[...,0]**2+gs0[...,1]**2)
gs1 = np.sqrt(gs1[...,0]**2+gs1[...,1]**2)
mesh.celldata['grad'] = gs0 
mesh.celldata['grad2'] = gs1 
fname = './netwon_'+ str(0).zfill(10) + '.vtu'
mesh.to_vtk(fname=fname)

