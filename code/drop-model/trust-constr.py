import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint, LinearConstraint
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from fealpy.mesh import IntervalMesh

class Model:
    def __init__(self, mesh, theta):
        self.mesh = mesh        
        self.theta = theta
        self.x0 = self.mesh.entity('node')  

    def __call__(self,x=None):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')
        mesh.node = mesh.entity('node') if x is None else x.reshape(NN, 2)  #更新节点
        node = mesh.node
        theta = self.theta    
        gamma = np.cos(theta)
        h = mesh.entity_measure('cell')
        a = np.sum(h[:-1])    #液体气体接触面积
        b = np.sum(h[-1])     #液体固体接触面积
        val = a - gamma*b     #能量函数
        return val

    def gradient(self, x=None):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')
        mesh.node = mesh.entity('node') if x is None else x.reshape(NN, 2)
        node = mesh.node
        theta = self.theta
        gamma = np.cos(theta)
        h = mesh.entity_measure('cell')
        v = mesh.cell_tangent()/h[:, np.newaxis]
        
        g = np.zeros((NN, 2), dtype=mesh.ftype)
        np.add.at(g, (cell[1:, 0], np.s_[:]), v[:-1])    #内部结点
        np.subtract.at(g, (cell[1:-1, 0], np.s_[:]), v[1:-1])
        
        g[0,0] = g[0,0] + gamma                          #端点
        g[-1]  = -v[0]
        g[-1,0] = g[-1,0] - gamma

        return g.reshape(-1)

    def volume(self, x=None):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')
        mesh.node = mesh.entity('node') if x is None else x.reshape(NN, 2)
        node = mesh.node
        n = mesh.cell_normal()
        a = np.sum(n*node[cell[:, 0]])/2.0
        return a


# 产生初始网格
n = 40
x = np.linspace(0, np.pi, n)
y = np.sin(x)
node = np.zeros((n, 2), dtype=np.float)
node[:, 0] = np.linspace(0, np.pi, n)
node[:, 1] = np.sin(node[:, 0])
cell = np.zeros((n, 2), dtype=np.int)
cell[:, 0] = range(n-1, -1, -1)
cell[:n-1, 1] = range(n-2, -1, -1)
cell[-1, 1] = n-1
mesh = IntervalMesh(node, cell)

#建立模型
theta = np.pi/6
model = Model(mesh, theta)
v = model.volume()
val = model()
grad = model.gradient().reshape(-1, 2)
NN = mesh.number_of_nodes()

fig = plt.figure()
plt.subplot(2,1,1)
node = model.x0
axes = fig.gca()
mesh.add_plot(axes)
axes.quiver(node[:, 0], node[:, 1], grad[:, 0], grad[:, 1])

#设置优化
def constraint1(x):
    return model.volume(x) - v

cons1 = {'type' : 'eq' , 'fun' : constraint1}
a = ((None,None),(None,None))
b = ((None,None),(0,0))
bnds = b + a*(NN-2) +b

re = minimize(fun = model,x0 = model.x0.reshape(-1),
               method='trust-constr', jac = model.gradient,
               constraints =cons1,bounds=bnds,options={'maxiter':1000,
                                                       'gtol': 1e-04 })

print(val,v)
print(re.fun,re.x,re.nit,re.message)

plt.subplot(2,1,2)
node = re.x.reshape(-1, 2) # 优化后的节点
grad = re.jac[0].reshape(-1,2)
axes = fig.gca()
mesh.add_plot(axes)
axes.quiver(node[:, 0], node[:, 1], grad[:, 0], grad[:, 1])
plt.show()
