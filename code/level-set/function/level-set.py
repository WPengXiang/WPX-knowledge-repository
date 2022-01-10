#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: level-set.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年12月06日 星期一 14时03分35秒
'''  
import numpy as np
class Level-set():
    def __inti__(self,s0,u,space):
        self.s0 = s0
        self.u = u
        self.space = space
    
    def SUPG(h): 
        n = space.function(dim=2)
        phi = space.basis(bcs)
        gphi = space.grad_basis(bcs)
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()
        s1 = space.function() 
        
        norm_u = integralalg.L2_norm(u,celltype=True)
        h = mesh.entity_measure('cell')
        s = h/(2*norm_u) 
        
        A1 = space.mass_matrix()
        
        E0 = np.einsum('i,ijk,ij,ijm,j -> jkm',ws,gphi[...,0],u(bcs)[...,0],phi,cellmeasure)
        E1 = np.einsum('i,ijk,ij,ijm,j -> jkm',ws,gphi[...,1],u(bcs)[...,1],phi,cellmeasure)
        E = E0+E1
        I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
        J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
        A2 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))
        
        E0 = np.einsum('i,ijk,ij,ijm,j,j -> jkm',\
                ws,gphi[...,0],u(bcs)[...,0],phi,cellmeasure,s)
        E1 = np.einsum('i,ijk,ij,ijm,j,j -> jkm',\
                ws,gphi[...,1],u(bcs)[...,1],phi,cellmeasure,s)
        E = E0+E1
        I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
        J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
        A2 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))
        
        E0 = np.einsum('i,ij,ijk,ijm,ij,j,j -> jkm',ws,u(bcs)[...,0],gphi[...,0]\
                ,gphi[...,0],u(bcs)[...,0],cellmeasure,s)
        E1 = np.einsum('i,ij,ijk,ijm,ij,j,j -> jkm',ws,u(bcs)[...,1],gphi[...,1]\
                ,gphi[...,1],u(bcs)[...,1],cellmeasure,s)
        E2 = np.einsum('i,ij,ijk,ijm,ij,j,j -> jkm',ws,u(bcs)[...,0],gphi[...,0]\
                ,gphi[...,1],u(bcs)[...,1],cellmeasure,s)
        E3 = np.einsum('i,ij,ijk,ijm,ij,j,j -> jkm',ws,u(bcs)[...,1],gphi[...,1]\
                ,gphi[...,0],u(bcs)[...,0],cellmeasure,s)
        E = E0+E1+E2+E3
        I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
        J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
        A4 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))

        A = A1 - dt/2*A2 + A3 + dt/2*A4 

        b = A1@s0 + dt/2*A2@s0 + A1@s0 - dt/2*A4@s0 
        
        x = spsolve(A,b)
        s1[:] = x[:]
        return s1

