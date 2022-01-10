#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: level-set.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年12月06日 星期一 14时03分35秒
'''  

from scipy.sparse import csr_matrix
from fealpy.decorator import cartesian,barycentric
from scipy.sparse.linalg import spsolve,cg
import numpy as np
class Level_set():
    def __init__(self,u,dt):
        self.u = u
        self.dt = dt
        self.dx = min(u.space.mesh.entity_measure('edge'))
        self.space = u.space 
        self.M = self.space.mass_matrix()
        self.S = self.space.stiff_matrix()
        self.CU = self.space.convection_matrix(c = u)
    
    def LS_CN_A(self):
        u = self.u
        dt = self.dt
        space = self.space
        s1 =  space.function()
        M = self.M
        CU= self.CU
        A = M + dt/2* CU
        return A
    
    def LS_CN_b(self,phi0):
        dt = self.dt
        space = self.space
        M = self.M
        C = self.CU
        b = M@phi0 - dt/2*C@phi0
        return b
   
    def CLS_SUPG_A(self):  
        u = self.u
        space = self.space
        mesh = space.mesh
        dt = self.dt
        qf = mesh.integrator(4,'cell')
        bcs,ws = qf.get_quadrature_points_and_weights()
        cellmeasure = mesh.entity_measure('cell')
 
        phi = space.basis(bcs)
        gphi = space.grad_basis(bcs)
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()
        
        s1 = space.function() 
        
        norm_u = np.sum(space.integralalg.L2_norm(u,celltype=True),axis=1)
        ss = self.dx/(2*norm_u) 
        M = self.M
        CU = self.CU
        
        @barycentric
        def s(bcs):
            val = np.einsum('ijm,j->ijm',u(bcs),ss)
            return val
        
        C2 = space.convection_matrix(c = s)
        self.CS = C2
        E0 = np.einsum('i,ij,ijk,ijm,ij,j -> jkm',\
                ws,s(bcs)[...,0],gphi[...,0],gphi[...,0],u(bcs)[...,0],cellmeasure)
        E1 = np.einsum('i,ij,ijk,ijm,ij,j -> jkm',\
                ws,s(bcs)[...,1],gphi[...,1],gphi[...,1],u(bcs)[...,1],cellmeasure)
        E2 = np.einsum('i,ij,ijk,ijm,ij,j -> jkm',\
                ws,s(bcs)[...,0],gphi[...,0],gphi[...,1],u(bcs)[...,1],cellmeasure)
        E3 = np.einsum('i,ij,ijk,ijm,ij,j -> jkm',\
                ws,s(bcs)[...,1],gphi[...,1],gphi[...,0],u(bcs)[...,0],cellmeasure)
        E = E0+E1+E2+E3
        I = np.broadcast_to(cell2dof[:,:,None],shape = E.shape)
        J = np.broadcast_to(cell2dof[:,None,:],shape = E.shape)
        A4 = csr_matrix((E.flat,(I.flat,J.flat)),shape=(gdof,gdof))
        self.SU = A4
        A = M - dt/2*CU + C2 + dt/2*A4  
        return A

    def CLS_SUPG_b(self,phi0):
        dt = self.dt
        space = self.space
        M = self.M
        CU = self.CU
        CS = self.CS
        SU = self.SU
        b = M@phi0 + dt/2*CU@phi0 + CS@phi0 - dt/2*SU@phi0
        return b
    
    def reinit_distance_function(self,s0,dt=0.0001,alpha=None):
        if alpha is None:
            alpha = 0.0625/self.dx
        space = self.space
        
        phi0 = s0
        phi1 = space.function()
        phi2 = space.function()
        phi1[:] = phi0


        S = self.S
        M = self.M
        while True:
            @barycentric
            def f(bcs):
                grad = phi1.grad_value(bcs)
                val = 1 - np.sqrt(np.sum(grad**2, -1))
                val *=  np.sign(phi0(bcs))
                return val

            b = M@phi1 + dt*space.source_vector(f) - dt*alpha*S@phi1
            phi2[:] = spsolve(M, b)

            error = space.integralalg.error(phi2, phi1)/dt
            print(error)
            if error <= 0.1:
                return phi2
            else:
                phi1[:] = phi2
'''
    def reinit_heavisisde(s,grads):
        print("开始重置")
        space = s.space
        phi = space.basis(bcs)
        gphi = space.grad_basis(bcs)
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()

        @barycentric
        def n(bcs):
            grad = grads(bcs)
            n = np.sqrt(np.sum(grad**2,-1))[...,np.newaxis]
            tag = grad==0
            n = grad/n
            n[tag] = 0
            return n
        error = 100
        
        M = space.mass_matrix()
        C = space.convection_matrix(c=n)
        A21 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
                ws,gphi[...,0],n(bcs)[...,0],gphi[...,0],n(bcs)[...,0],cellmeasure)
        A22 = np.einsum('i,ijk,ij,ijm,ij,j-> jkm',\
                ws,gphi[...,1],n(bcs)[...,1],gphi[...,1],n(bcs)[...,1],cellmeasure)
        A23 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
                ws,gphi[...,1],n(bcs)[...,1],gphi[...,0],n(bcs)[...,0],cellmeasure)
        A24 = np.einsum('i,ijk,ij,ijm,ij,j -> jkm',\
                ws,gphi[...,0],n(bcs)[...,0],gphi[...,1],n(bcs)[...,1],cellmeasure)
        A2 = A21+A22+A23+A24
        I = np.broadcast_to(cell2dof[:,:,None],shape = A2.shape)
        J = np.broadcast_to(cell2dof[:,None,:],shape = A2.shape)
        A2 = csr_matrix((A2.flat,(I.flat,J.flat)),shape=(gdof,gdof))
        while True:         
            @barycentric
            def fun(bcs):
                val = np.einsum('ijk,ij->ijk',n(bcs),s(bcs))
                return val
            
            C1 = space.convection_matrix(c=fun)
            
            A = M - (dt/2)*C + (epsilon*dt/2)*A2 + dt*C1
             
            b = M@s + (dt/2)*C*s - (epsilon*dt/2)*A2@s

            x = spsolve(A,b)

            errorold = error
            sold = s
            error = space.function()
            error[:]=x[:]-s[:]
            error = integralalg.L2_norm(error)/dt
            s[:] = x[:]
            print(error)
            if error>=errorold or error <=0.01 :
                return s
                break


    def grads(self):
        epsilon = self.epsilon

        phi = space.basis(bcs)
        gphi = s.grad_value(bcs)
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()

        A = space.mass_matrix()
        
        s11 = s.grad_value(bcs)[...,0]
        b11 = np.einsum('i,ijk,ij,j -> jk',ws,phi,s11,cellmeasure)
        b1 = np.zeros(gdof)
        np.add.at(b1,cell2dof,b11)
        s12 = s.grad_value(bcs)[...,1]
        b12 = np.einsum('i,ijk,ij,j -> jk',ws,phi,s12,cellmeasure)
        b2 = np.zeros(gdof)
        np.add.at(b2,cell2dof,b12)

        x1 = spsolve(A,b1)
        x2 = spsolve(A,b2)

        grads = space.function(dim=2)
        grads[:,0] = x1[:]
        grads[:,1] = x2[:]
        return grads

    def area(s,space):
        phi = space.basis(bcs)
        gphi = space.grad_basis(bcs)
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()
        
        fun = space.function()
        fun[:] = 1/2*(np.abs(0.5-s[:])/(0.5-s[:])+1)
        value = space.integralalg.integral(fun)
        return value
'''
