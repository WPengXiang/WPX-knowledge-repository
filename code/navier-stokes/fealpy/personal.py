#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: personal.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年10月25日 星期一 19时23分31秒
'''  
import numpy as np
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from scipy.sparse import csr_matrix,hstack,vstack,spdiags

domain = [0, 1, 0, 1]
smesh = MF.boxmesh2d(domain, nx=2, ny=2, meshtype='tri')
uspace = LagrangeFiniteElementSpace(smesh, p=1)
M = uspace.mass_matrix()
print(M.toarray().sum())

def integral_matrix(q,fun0,fun1):
    mesh = smesh
    space0 = uspace
    
    qf = mesh.integrator(4,'cell')
    bcs,ws = qf.get_quadrature_points_and_weights()
    cellmeasure = mesh.entity_measure('cell')
    cell2dof = space0.cell_to_dof()
    gdof = space0.number_of_global_dofs()

    A = np.einsum('i,ijk,ijm,j->jkm',ws,fun0,fun1,cellmeasure)
    I = np.broadcast_to(cell2dof[:,:,None],shape = A.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape = A.shape)
    A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(gdof,gdof))

    return A


qf = smesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = smesh.entity_measure('cell')

## 速度空间
phi = uspace.basis(bcs)
gphi = uspace.grad_basis(bcs)
gdof = uspace.number_of_global_dofs()
cell2dof = uspace.cell_to_dof()

A = integral_matrix(4,phi,phi)
print(A.sum())
def nolinear_matrix(uh, q=3):

    space = uh.space
    mesh = space.mesh

    qf = mesh.integrator(q, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()

    cellmeasure = mesh.entity_measure('cell')

    cval = 2*uh(bcs)[...,None]*uh.grad_value(bcs) # (NQ, NC, GD)
    phi = space.basis(bcs)       # (NQ, 1, ldof)
    gphi = space.grad_basis(bcs) # (NQ, NC, ldof, GD)

    B = np.einsum('q, qcid, qcd, qcj, c->cij', ws, gphi, cval, phi, cellmeasure)

    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    I = np.broadcast_to(cell2dof[:, :, None], shape=B.shape)
    J = np.broadcast_to(cell2dof[:, None, :], shape=B.shape)
    B = csr_matrix((B.flat,(I.flat,J.flat)), shape=(gdof,gdof))

