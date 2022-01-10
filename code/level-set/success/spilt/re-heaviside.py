#!/usr/bin/python3
'''    	
@ Author: wpx
@ File Name: level.py
@ Author: wpx
@ Mail: wpx15673207315@gmail.com 
@ Created Time: 2021年11月19日 星期五 11时42分52秒
'''  
import argparse 
import numpy as np

from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.decorator import cartesian,barycentric

from fealpy.decorator import cartesian, barycentric
from scipy.sparse.linalg import spsolve,cg

from scipy.sparse import bmat,csr_matrix,hstack,vstack,spdiags
#参数解析
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

parser.add_argument('--outputdir',
        default='~/result/', type=str,
        help='')

args = parser.parse_args()

dim = args.dim
degree = args.degree
nt = args.nt
ns = args.ns
T = args.T


@cartesian
def u(p):
    x = p[..., 0]
    y = p[..., 1]
    u = np.zeros(p.shape)
    u[..., 0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[..., 1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u

@cartesian
def circle(p):
    x = p[...,0]
    y = p[...,1]
    val = np.sqrt((x-0.5)**2+(y-0.75)**2)-0.15
    val = 1/(1+np.exp(-val/epsilon)) 
    return val

@cartesian
def sphere(p):
    x = p[...,0]
    y = p[...,1]
    z = p[...,2]
    val = np.sqrt((x-0.5)**2+(x-0.5)**2+(y-0.075)**2)-0.15
    return val


if dim == 2:
    domain = [0, 1, 0, 1]
    mesh = MF.boxmesh2d(domain, nx=ns, ny=ns, meshtype='tri')
else:
    domain = [0, 1, 0, 1, 0, 1]
    mesh = MF.boxmesh3d(domain, nx=ns, ny=ns, nz=ns, meshtype='tet')

timeline = UniformTimeLine(0,T,nt)
mesh = MF.boxmesh2d(domain,nx=ns,ny=ns,meshtype='tri')
space = LagrangeFiniteElementSpace(mesh,p=1)
dx = min(mesh.entity_measure('edge'))
epsilon = (dx**0.9)/2 

'''
fig1 = plt.figure()
node = mesh.node
x = tuple(node[:,0])
y = tuple(node[:,1])
NN = mesh.number_of_nodes()
u = u(node)[:NN]
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


integralalg = space.integralalg

u = space.interpolation(u)
s0 = space.interpolation(circle)

dt = timeline.dt

qf = mesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')

def SUPG(u,s0): 
    space = s0.space
    mesh = space.mesh

    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    
    s1 = space.function() 
    
    norm_u = np.sum(integralalg.L2_norm(u,celltype=True),axis=1)
    h = min(mesh.entity_measure('edge'))
    ss = h/(2*norm_u) 
    
    @barycentric
    def s(bcs):
        val = np.einsum('ijm,j->ijm',u(bcs),ss)
        return val
    M = space.mass_matrix()
    C1 = space.convection_matrix(c = u)
    C2 = space.convection_matrix(c = s)
    
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

    A = M - dt/2*C1 + C2 + dt/2*A4 

    b = M@s0 + dt/2*C1@s0 + C2@s0 - dt/2*A4@s0 
    
    x = spsolve(A,b)
    s1[:] = x[:]
    return s1

def grads(s,space):
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

def re(s,grads):
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

def area(s,space):
    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    
    fun = space.function()
    fun[:] = 1/2*(np.abs(0.5-s[:])/(0.5-s[:])+1)
    value = space.integralalg.integral(fun)
    return value
    
    

fname = './test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['uh'] = s0
mesh.to_vtk(fname=fname)

for i in range(0,nt):
    t1 = timeline.next_time_level()
    if np.abs(t1- T/2)<1e-10:
        u[:] = -u
    print("t1=",t1)
    print("面积:",area(s0,space))
    s0 = SUPG(u,s0)
    gra = grads(s0,space)
    s0 = re(s0,gra)
    mesh.nodedata['uh'] = s0
    mesh.nodedata['u'] = u
    #mesh.celldata['grad'] = norm_s
    fname = './test_'+ str(i+1).zfill(10) + '.vtu'
    mesh.to_vtk(fname=fname)
    timeline.advance()

