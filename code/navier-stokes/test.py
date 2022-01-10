#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: test.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年08月03日 星期二 19时42分31秒
'''  
import numpy as np
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import VectorLagrangeFiniteElementSpace
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

domain = [0,1,0,1]
mesh  = MF.boxmesh2d(box=domain,nx=2,ny=2,meshtype='tri')
space = LagrangeFiniteElementSpace(mesh,p=2)

qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')
ipoint = space.interpolation_points()


phi = space.basis(bcs)
A = np.einsum('i,ijk,ijm,j -> jkm',ws,phi,phi,cellmeasure)
gdof = space.number_of_global_dofs()
cell2dof = space.cell_to_dof()
I = np.broadcast_to(cell2dof[:,:,None],shape=A.shape)
J = np.broadcast_to(cell2dof[:,None,:],shape=A.shape)
H = np.zeros((gdof,gdof))
np.add.at(H,(I,J),A)

gphi = space.grad_basis(bcs)
E0 = np.einsum('i,ijk,ijm,j -> jkm',ws,gphi[...,0],gphi[...,0],cellmeasure)
E1 = np.einsum('i,ijk,ijm,j -> jkm',ws,gphi[...,1],gphi[...,1],cellmeasure)
E2 = np.einsum('i,ijk,ijm,j -> jkm',ws,gphi[...,1],gphi[...,0],cellmeasure)
E00 = np.zeros((gdof,gdof))
E11 = np.zeros((gdof,gdof))
E01 = np.zeros((gdof,gdof))
np.add.at(E00,(I,J),E0+1/2*E1)
np.add.at(E11,(I,J),E1+1/2*E0)
np.add.at(E01,(I,J),1/2*E2)

H11 = np.zeros((gdof,gdof))
H1 = np.hstack((H,H11))
H2 = np.hstack((H11,H))
HH = np.vstack((H1,H2))
EE0 = np.hstack((E00,E01))
EE1 = np.hstack((E01,E11))
EE = np.vstack((EE0,EE1)) 
#print(EE)

G0 =  np.einsum('i,ijk,ijm,ijk,j -> jkm',ws,phi,phi,gphi[...,0],cellmeasure)
G1 =  np.einsum('i,ijk,ijm,ijk,j -> jkm',ws,phi,phi,gphi[...,1],cellmeasure)
G00 = np.zeros((gdof,gdof))
G11 = np.zeros((gdof,gdof))
np.add.at(G00,(I,J),G0)
np.add.at(G11,(I,J),G1)
GG1 = np.zeros((gdof,gdof))
GG2 = np.hstack((G00,GG1))
GG3 = np.hstack((GG1,G11))
GG = np.vstack((GG2,GG3))
#print(GG)
uh = space.function(dim=2)
b = uh.flatten(order='F')

A = 5*HH+1/2*EE
#print(A)
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes,node=ipoint,showindex=True)
mesh.find_edge(axes,showindex=False)
mesh.find_cell(axes,showindex=True)
plt.show()
