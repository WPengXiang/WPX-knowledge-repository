import argparse
import sys
import numpy as np
import matplotlib

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from mumps import DMumpsContext

import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.boundarycondition import DirichletBC 
from fealpy.timeintegratoralg import UniformTimeLine
## Stokes model
from navier_stokes_mold_2d import Poisuille as PDE

## error anlysis tool
from fealpy.tools import showmultirate

# 参数设置
udegree = 2
pdegree = 1
ns = 8
nt = 100
T = 10
rho = 1
mu=1
udim = 2

pde = PDE()
smesh = MF.boxmesh2d(pde.domain(), nx=ns, ny=ns, meshtype='tri')
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt

uspace = LagrangeFiniteElementSpace(smesh,p=udegree)
pspace = LagrangeFiniteElementSpace(smesh,p=pdegree)

u0 = uspace.function(dim=udim)
u1 = uspace.function(dim=udim)

p1 = pspace.function()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = pgdof+2*ugdof

A = uspace.stiff_matrix()
B1,B2 = uspace.div_matrix(pspace)

C = pspace.mass_matrix()
E = (1/dt)*uspace.mass_matrix()

#ctx = DMumpsContext()
#ctx.set_silent()
errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(0,nt):

    # 下一个时间层t1
    t1 = tmesh.next_time_level()
    print("t1=",t1)

    #左端项
    
    D1,D2 = uspace.div_matrix(uspace)
    D = D1 * np.broadcast_to(u0[...,0],D1.shape)+\
        D2 * np.broadcast_to(u0[...,1],D1.shape) 
    
    M = bmat([[E+A+D,None,-B1],[None,E+A+D,-B2],[-B1.T,-B2.T,None]],format='csr')
    
    #右端项
    F = uspace.source_vector(pde.source,dim=udim) + E@u0
    FF = np.r_['0', F.T.flat, np.zeros(pgdof)]

    u_isBdDof = uspace.is_boundary_dof()
    #p_isBdDof = np.zeros(pgdof,dtype=np.bool)
    p_isBdDof = pspace.is_boundary_dof(threshold=pde.is_p_boundary)
    
    x = np.zeros(gdof,np.float)
    ipoint = uspace.interpolation_points()
    uso = pde.u_dirichlet(ipoint)
    x[0:ugdof][u_isBdDof] = uso[:,0][u_isBdDof]
    x[ugdof:2*ugdof][u_isBdDof] = uso[u_isBdDof][:,1]
    ipoint = pspace.interpolation_points()
    pso = pde.p_dirichlet(ipoint)
    x[-pgdof:][p_isBdDof] = pso[p_isBdDof]

    isBdDof = np.hstack([u_isBdDof, u_isBdDof, p_isBdDof])
    
    FF -= M@x
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    M = T@M@T + Tbd
    FF[isBdDof] = x[isBdDof]

    x[:] = spsolve(M, FF)
    u1[:, 0] = x[:ugdof]
    u1[:, 1] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
    
    uc1 = pde.velocity(smesh.node)
    NN = smesh.number_of_nodes()
    uc2 = u1[:NN]
    up1 = pde.pressure(smesh.node)
    up2 = p1[:NN]
   
    errorMatrix[0,i] = uspace.integralalg.L2_error(pde.velocity,u1)
    errorMatrix[1,i] = pspace.integralalg.error(pde.pressure,p1)
    errorMatrix[2,i] = np.abs(uc1-uc2).max()
    errorMatrix[3,i] = np.abs(up1-up2).max()

    u0[:] = u1 

    tmesh.advance()

print("uL2:",errorMatrix[2,-1])
print("pL2:",errorMatrix[1,-1])
print("umax:",errorMatrix[2,-1])
print("pmax:",errorMatrix[3,-1])
'''
fig1 = plt.figure()
node = smesh.node
x = tuple(node[:,0])
y = tuple(node[:,1])
NN = smesh.number_of_nodes()
u = u1[:NN]
ux = tuple(u[:,0])
uy = tuple(u[:,1])

o = ux
norm = matplotlib.colors.Normalize()
cm = matplotlib.cm.copper
sm = matplotlib.cm.ScalarMappable(cmap=cm,norm=norm)
sm.set_array([])
plt.quiver(x,y,ux,uy,color=cm(norm(o)))
plt.colorbar(sm)
plt.show()

'''
