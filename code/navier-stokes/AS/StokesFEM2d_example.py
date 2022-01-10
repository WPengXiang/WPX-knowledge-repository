
import argparse
import sys
import numpy as np

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve

from mumps import DMumpsContext
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.boundarycondition import DirichletBC 
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
## Stokes model ,8
from fealpy.pde.stokes_model_2d import StokesModelData_0 as PDE

## error anlysis tool
from fealpy.tools import showmultirate

degree = 2
dim = 2
maxit = 4

pde = PDE()
#mesh = MF.boxmesh2d(pde.box, nx=5, ny=5, meshtype='tri')
mesh = pde.init_mesh(n=3)
errorType = ['$|| u - u_h||_0$',
             '$|| p - p_h||_0$'
             ]

errorMatrix = np.zeros((2, maxit), dtype=np.float64)
NDof = np.zeros(maxit, dtype=np.int_)
'''
ctx = DMumpsContext()
ctx.set_silent()
'''
for i in range(maxit):
    print("The {}-th computation:".format(i))

    uspace = LagrangeFiniteElementSpace(mesh, p=degree)
    pspace = LagrangeFiniteElementSpace(mesh, p=degree-1)

    ugdof = uspace.number_of_global_dofs()
    pgdof = pspace.number_of_global_dofs()

    uh = uspace.interpolation(pde.velocity)
    ph = pspace.function()

    qf = mesh.integrator(4,'cell')        
    ipoint = uspace.interpolation_points()  
    bcs,ws = qf.get_quadrature_points_and_weights() 
    cellmeasure = mesh.entity_measure('cell')      

    ## 速度空间
    uphi = uspace.basis(bcs) 
    ugphi = uspace.grad_basis(bcs)
    ugdof = uspace.number_of_global_dofs()  
    ucell2dof = uspace.cell_to_dof()   

    ## 压力空间
    pphi = pspace.basis(bcs) 
    pgphi = pspace.grad_basis(bcs)
    pgdof = pspace.number_of_global_dofs()
    pcell2dof = pspace.cell_to_dof()


    A0 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],ugphi[...,0],cellmeasure)  
    A1 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],ugphi[...,1],cellmeasure)  
    I = np.broadcast_to(ucell2dof[:,:,None],shape = A0.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape = A0.shape)
    A = csr_matrix(((A0+A1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))

    B1 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],pphi,cellmeasure)  
    B2 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],pphi,cellmeasure) 
    I = np.broadcast_to(ucell2dof[:,:,None],shape = B1.shape)
    J = np.broadcast_to(pcell2dof[:,None,:],shape =B2.shape)
    B1 = csr_matrix(((B1).flat,(I.flat,J.flat)),shape=(ugdof,pgdof))
    B2 = csr_matrix(((B2).flat,(I.flat,J.flat)),shape=(ugdof,pgdof))

    #组装第3个矩阵（div（u^{n}）,q）
    C1 = B1.T
    C2 = B2.T
    
    HH = np.einsum('i,ijk,ijm,j -> jkm',ws,pphi,pphi,cellmeasure) # （NC，ldof，ldof）
    I = np.broadcast_to(pcell2dof[:,:,None],shape = HH.shape)
    J = np.broadcast_to(pcell2dof[:,None,:],shape = HH.shape)
    H = csr_matrix((HH.flat,(I.flat,J.flat)),shape=(pgdof,pgdof)) #

    '''
    A = uspace.stiff_matrix()
    B0, B1 = uspace.div_matrix(pspace)
    F = uspace.source_vector(pde.source, dim=dim)    
    C = pspace.mass_matrix()
    '''

    M = bmat([[A,None,-B1],[None,A,-B2],[C1,C2,10e-8*H]])
    
    isuBDof = uspace.boundary_dof()
    ispBDof = pspace.boundary_dof()
    isBdDof = np.block([isuBDof, isuBDof, ispBDof])
    bdIdx = np.zeros((M.shape[0]),dtype = float)           
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx,0,M.shape[0],M.shape[0])     
    T = spdiags(1-bdIdx,0,M.shape[0],M.shape[0])
    M = T@M+ Tbd
   
    #计算载荷向量,并处理边界
    b = uspace.source_vector(pde.source,dim=dim)
    b = np.r_['0', b.T.flat, np.zeros(pgdof)]
    ipoint = uspace.interpolation_points()
    uso = pde.dirichlet(ipoint)
    pipoint = pspace.interpolation_points()
    pso = pde.pressure(pipoint)
    b[0:ugdof][isuBDof] = uso[:,0][isuBDof]
    b[ugdof:2*ugdof][isuBDof] = uso[isuBDof][:,1]
    b[2*ugdof:][ispBDof] = pso[ispBDof]
     
    fus= spsolve(M,b)
    uh[:, 0] = fus[:ugdof]
    uh[:, 1] = fus[ugdof:2*ugdof]
    ph[:] = fus[2*ugdof:]
  
 

    '''
    FF = np.r_['0', F.T.flat, np.zeros(pgdof)]
    isBdDof = uspace.is_boundary_dof()
    gdof = 2*ugdof + pgdof
    x = np.zeros(gdof,np.float)
    ipoint = uspace.interpolation_points()
    uso = pde.dirichlet(ipoint)
    x[0:ugdof][isBdDof] = uso[:,0][isBdDof]
    x[ugdof:2*ugdof][isBdDof] = uso[isBdDof][:,1]
   
    isBdDof = np.block([isBdDof, isBdDof, np.zeros(pgdof, dtype=np.bool)])

    FF -= AA@x
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    AA = T@AA@T + Tbd
    FF[isBdDof] = x[isBdDof]
    
    ctx.set_centralized_sparse(AA)
    xx = FF.copy()
    ctx.set_rhs(xx)
    ctx.run(job=6)
    uh[:, 0] = xx[:ugdof]
    uh[:, 1] = xx[ugdof:2*ugdof]
    ph[:] = xx[2*ugdof:]
    '''
    NDof[i] =  2*ugdof+pgdof 

    errorMatrix[0, i] = uspace.integralalg.error(pde.velocity, uh)
    errorMatrix[1, i] = pspace.integralalg.error(pde.pressure, ph)
    if i < maxit-1:
        mesh.uniform_refine()
    
print(errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix, errorType)
plt.show()


























