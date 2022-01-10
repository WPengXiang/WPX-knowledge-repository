import numpy as np

import scipy 
import math
import matplotlib.pyplot as plt
import matplotlib

from  fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric
from fealpy.boundarycondition import DirichletBC

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat

degree = 2                               #基函数次数
dim = 2
ns = 18                                  #空间离散
eps = 1e-8
             


@cartesian
def is_p_boundary(p):
    return (np.abs(p[..., 0]) <1e-8) | (np.abs(p[..., 0] - 1) <1e-8) |(np.abs(p[..., 1]) < 1e-8) | (np.abs(p[..., 1] - 1) < 1e-8)


@cartesian
def is_v_boundary(p):
    return (np.abs(p[..., 0]) <1e-8) | (np.abs(p[..., 0] - 1) <1e-8) |(np.abs(p[..., 1]) < 1e-8) | (np.abs(p[..., 1] - 1) < 1e-8)
@cartesian
def velocity(p):
    x =p[...,0]
    y =p[...,1]
    u = np.zeros(p.shape,dtype=np.float)
    u[...,0] =  10*x**2*(x-1)**2*y*(y-1)*(2*y-1)
    u[...,1] =  -10*x*(x-1)*(2*x-1)*y**2*(y-1)**2
    return u    
@cartesian      
def pressure(p):         
    x = p[..., 0]         
    y = p[..., 1]         
    pre = 10*(2*x-1)*(2*y-1)
    return pre
@cartesian
def source(p):        
    x = p[..., 0]         
    y = p[..., 1]                 
    f = np.zeros(p.shape,dtype=np.float)         
    f[...,0] =  (1*(-120*x**4*y + 60*x**4 + 240*x**3*y - 120*x**3 - 240*x**2*y**3 + 360*x**2*y**2 - 240*x**2*y + 60*x**2 + 240*x*y**3 - 360*x*y**2 + 120*x*y - 40*y**3 + 60*y**2 - 20*y) + (40*y - 20) + (-100*x**3*y**2*(x - 1)**3*(2*x - 1)*(y - 1)**2*(2*y*(y - 1) + y*(2*y - 1) + (y - 1)*(2*y - 1) - 2*(2*y - 1)**2)))                         
    f[...,1] =  (1*(240*x**3*y**2 - 240*x**3*y + 40*x**3 - 360*x**2*y**2 + 360*x**2*y - 60*x**2 + 120*x*y**4 - 240*x*y**3 + 240*x*y**2 - 120*x*y + 20*x - 60*y**4 + 120*y**3 - 60*y**2) + (40*x - 20) + (-100*x**2*y**3*(x - 1)**2*(y - 1)**3*(2*y - 1)*(2*x*(x - 1) + x*(2*x - 1) + (x - 1)*(2*x - 1) - 2*(2*x - 1)**2)))        
    return f 
@cartesian 
def source1(p):        
    x = p[..., 0]         
    y = p[..., 1]                 
    #f = np.zeros(p.shape,dtype=np.float)         
    f =  (1*(-120*x**4*y + 60*x**4 + 240*x**3*y - 120*x**3 - 240*x**2*y**3 + 360*x**2*y**2 - 240*x**2*y + 60*x**2 + 240*x*y**3 - 360*x*y**2 + 120*x*y - 40*y**3 + 60*y**2 - 20*y) + (40*y - 20) + (-100*x**3*y**2*(x - 1)**3*(2*x - 1)*(y - 1)**2*(2*y*(y - 1) + y*(2*y - 1) + (y - 1)*(2*y - 1) - 2*(2*y - 1)**2)))                         
#    f[...,1] =  (1*(240*x**3*y**2 - 240*x**3*y + 40*x**3 - 360*x**2*y**2 + 360*x**2*y - 60*x**2 + 120*x*y**4 - 240*x*y**3 + 240*x*y**2 - 120*x*y + 20*x - 60*y**4 + 120*y**3 - 60*y**2) + (40*x - 20) + (-100*x**2*y**3*(x - 1)**2*(y - 1)**3*(2*y - 1)*(2*x*(x - 1) + x*(2*x - 1) + (x - 1)*(2*x - 1) - 2*(2*x - 1)**2)))        
    return f 
@cartesian 
def source2(p):        
    x = p[..., 0]         
    y = p[..., 1]                 
    #f = np.zeros(p.shape,dtype=np.float)         
#    f =  (1*(-120*x**4*y + 60*x**4 + 240*x**3*y - 120*x**3 - 240*x**2*y**3 + 360*x**2*y**2 - 240*x**2*y + 60*x**2 + 240*x*y**3 - 360*x*y**2 + 120*x*y - 40*y**3 + 60*y**2 - 20*y) + (40*y - 20) + (-100*x**3*y**2*(x - 1)**3*(2*x - 1)*(y - 1)**2*(2*y*(y - 1) + y*(2*y - 1) + (y - 1)*(2*y - 1) - 2*(2*y - 1)**2)))                         
    f =  (1*(240*x**3*y**2 - 240*x**3*y + 40*x**3 - 360*x**2*y**2 + 360*x**2*y - 60*x**2 + 120*x*y**4 - 240*x*y**3 + 240*x*y**2 - 120*x*y + 20*x - 60*y**4 + 120*y**3 - 60*y**2) + (40*x - 20) + (-100*x**2*y**3*(x - 1)**2*(y - 1)**3*(2*y - 1)*(2*x*(x - 1) + x*(2*x - 1) + (x - 1)*(2*x - 1) - 2*(2*x - 1)**2)))        
    return f 
domain = [0, 1, 0, 1]
smesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
uspace = LagrangeFiniteElementSpace(smesh, p=degree) 
pspace = LagrangeFiniteElementSpace(smesh, p=degree-1)

u0 = uspace.function(dim=2)
u1 = uspace.function(dim=2) #u^{n-1}
p1 = pspace.function() 

qf = smesh.integrator(4,'cell')        
ipoint = uspace.interpolation_points()  
bcs,ws = qf.get_quadrature_points_and_weights() 
cellmeasure = smesh.entity_measure('cell')      

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

##组装第0个矩阵即为（u^{n},v） ############(应该没错，注意行列)#####################
HH = np.einsum('i,ijk,ijm,j -> jkm',ws,pphi,pphi,cellmeasure) # （NC，ldof，ldof）
I = np.broadcast_to(pcell2dof[:,:,None],shape = HH.shape)
J = np.broadcast_to(pcell2dof[:,None,:],shape = HH.shape)
H = csr_matrix((HH.flat,(I.flat,J.flat)),shape=(pgdof,pgdof)) #



##组装第1个矩阵即为（gradient(u^{n}),gradient(v)）
A0 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],ugphi[...,0],cellmeasure)  
A1 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],ugphi[...,1],cellmeasure)  
I = np.broadcast_to(ucell2dof[:,:,None],shape = A0.shape)
J = np.broadcast_to(ucell2dof[:,None,:],shape = A0.shape)
A = csr_matrix(((A0+A1).flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
#A = csr_matrix(scipy.linalg.block_diag(A00.toarray(),A00.toarray(),H0))# (3267, 3267)

#组装第2个矩阵（div（v^{n}）,p） 
B1 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,0],pphi,cellmeasure)  
B2 = np.einsum('i,ijk,ijm,j -> jkm',ws,ugphi[...,1],pphi,cellmeasure) 
I = np.broadcast_to(ucell2dof[:,:,None],shape = B1.shape)
J = np.broadcast_to(pcell2dof[:,None,:],shape =B2.shape)
B1 = csr_matrix(((B1).flat,(I.flat,J.flat)),shape=(ugdof,pgdof))
B2 = csr_matrix(((B2).flat,(I.flat,J.flat)),shape=(ugdof,pgdof))

#组装第3个矩阵（div（u^{n}）,q）
C1 = B1.T
C2 = B2.T


##组装第0个矩阵即为（u^{n},v） ############(应该没错，注意行列)#####################
HH = np.einsum('i,ijk,ijm,j -> jkm',ws,pphi,pphi,cellmeasure) # （NC，ldof，ldof）
I = np.broadcast_to(pcell2dof[:,:,None],shape = HH.shape)
J = np.broadcast_to(pcell2dof[:,None,:],shape = HH.shape)
H = csr_matrix((HH.flat,(I.flat,J.flat)),shape=(pgdof,pgdof)) #


iter = 0
while 1:#(0,nt): 
    iter =iter +1
    #组装第4个矩阵（(u^{n-1}.gradient)u^{n},v）  $$$$$$这一部分需要放到迭代里面
    D = np.einsum('i,ijk,ijl,ijml,j -> jkm',ws,uphi,u0(bcs),ugphi,cellmeasure)
#    D2 = np.einsum('i,ijk,ijn,ijm,j -> jkm',ws,ugphi[...,1],u0(bcs)[...,1][...,None],uphi,cellmeasure)
    I = np.broadcast_to(ucell2dof[:,:,None],shape = D.shape)
    J = np.broadcast_to(ucell2dof[:,None,:],shape =D.shape)
    D = csr_matrix((D.flat,(I.flat,J.flat)),shape=(ugdof,ugdof))
#    G = np.zeros((pgdof,pgdof))
#    G[0,0] =2
    
    
    
    
    #总左端矩阵
    M = bmat([[A+D,None,-B1],[None,A+D,-B2],[C1,C2,10e-8*H]])


    ##矩阵边界处理
    isBDof1 = uspace.boundary_dof(threshold = is_v_boundary)
    isBDof2 = np.zeros(pgdof,dtype = bool)
    isBDof = np.hstack((isBDof1,isBDof1,isBDof2))               
    bdIdx = np.zeros((M.shape[0]),dtype = float)           
    bdIdx[isBDof] = 1
    Tbd = spdiags(bdIdx,0,M.shape[0],M.shape[0])     
    T = spdiags(1-bdIdx,0,M.shape[0],M.shape[0])
    M = T@M+ Tbd
    #计算载荷向量,并处理边界
    b1 = uspace.source_vector(source1)
    b2 = uspace.source_vector(source2) 
    b =  np.hstack((b1,b2))
    fisbdf = uspace.boundary_dof(threshold = is_v_boundary)
    fisbdf = np.hstack((fisbdf,fisbdf))
    b[fisbdf] = 0
    
    
    #处理p部分的载荷向量
    b3 = np.zeros(pgdof)
#    b3[0] = 1  ####################%%%%%%%%%%%%%%%%%%%%%
    b = np.hstack((b,b3)) 

     
    fus= spsolve(M,b)
    u1[:,0] = fus[0:ugdof]
    u1[:,1] = fus[ugdof:2*ugdof]
    p1[:] = fus[2*ugdof:]
#    p1[0] = 0.5
    
    err = np.max(np.abs(u1 - u0))
    print("迭代次数：#####################3",iter)
    uL2 = uspace.integralalg.L2_error(velocity, u1)
    print('uL2',uL2)
    
   
    pL2 = pspace.integralalg.L2_error(pressure, p1)
    print('pL2',pL2)
#    

#     #H1 = uspace.integralalg.L2_error(gradient, u0.grad_value)    
#    if err <1e-8 :
#        break
    u0[:] = u1[:]
    if iter>10:
        break
#L2 = uspace.integralalg.L2_error(velocity, u0)
#H1 = uspace.integralalg.L2_error(gradient, u0.grad_value)

#print('L2',L2)
# #画图
#plt.figure()
#node = smesh.node
#x = tuple(node[:,0])
#y = tuple(node[:,1])
#NN = smesh.number_of_nodes()
#u = u0[:NN]
#ux = tuple(u[:,0])
#uy = tuple(u[:,1])
#
#o = ux
#norm = matplotlib.colors.Normalize()
#cm = matplotlib.cm.copper
#sm = matplotlib.cm.ScalarMappable(cmap=cm,norm=norm)
#sm.set_array([])
#plt.quiver(x,y,ux,uy,color=cm(norm(o)))
#plt.colorbar(sm)
#plt.show()

#def velocity(p):         """ the exact solution         """        
#    x = p[...,0]
#    y = p[...,1]         
#    u = np.zeros(p.shape,dtype=np.float)         
#    u[...,0] =  10*x**2*(x-1)**2*y*(y-1)*(2*y-1)         
#    u[...,1] =  -10*x*(x-1)*(2*x-1)*y**2*(y-1)**2                  
#    return u   
       
#def gradient(self,p):         """ The gradient of the exact solution         """         
#    x = p[..., 0]         
#    y = p[..., 1]         
#    pp = len(p.shape)         
#    if pp==1:             
#        shape = (2,2)         
#    elif pp==2:             
#        shape = (p.shape[0],2,2)         
#    elif pp==3:             
#        shape = (p.shape[0],p.shape[1],2,2)         
#    else:             
#        raise ValueError("No shape!")         
#    val = np.zeros(shape,dtype=np.float)         
#    val[..., 0,0] = (20*x*y*(x - 1)*(2*x - 1)*(y - 1)*(2*y - 1))         
#    val[..., 0,1] = (10*x**2*(x - 1)**2*(2*y*(y - 1) + y*(2*y - 1) + (y - 1)*(2*y - 1)))                  
#    val[..., 1,0] = (-10*y**2*(y - 1)**2*(2*x*(x - 1) + x*(2*x - 1) + (x - 1)*(2*x - 1)))         
#    val[..., 1,1] = (20*x*y*(1 - 2*y)*(x - 1)*(2*x - 1)*(y - 1))                  
#        return val 


































































































