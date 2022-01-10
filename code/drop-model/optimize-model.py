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
        bc = mesh.entity_barycenter('cell')
        self.flag = (bc[:, 1] == 0)   

    def __call__(self, x=None):
        mesh = self.mesh
        flag = self.flag
        NN = mesh.number_of_nodes()
        node = mesh.entity('node') if x is None else x.reshape(NN, 2)
        theta = self.theta
        gamma = np.cos(theta)
        h = mesh.entity_measure('cell', node=node)
        a = np.sum(h[~flag])
        b = np.sum(h[flag])
        val = a - gamma*b
        return val

    def gradient(self, x=None):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        flag = self.flag
        cell = mesh.entity('cell')
        node = mesh.entity('node') if x is None else x.reshape(NN, 2)
        theta = self.theta
        gamma = np.cos(theta)
        h = mesh.entity_measure('cell', node=node)
        v = mesh.cell_tangent(node=node)/h[:, np.newaxis]
        g = np.zeros((NN, 2), dtype=mesh.ftype)
        np.add.at(g, (cell[~flag, 1], np.s_[:]), v[~flag])
        np.subtract.at(g, (cell[~flag, 0], np.s_[:]), v[~flag])

        np.subtract.at(g, (cell[flag, 1], np.s_[:]), gamma*v[flag])
        np.add.at(g, (cell[flag, 0], np.s_[:]), gamma*v[flag])

        return g.reshape(-1)

    def hessian(self, x=None):
        mesh = self.mesh
        

    def volume(self, x=None):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        cell = mesh.entity('cell')
        node = mesh.entity('node') if x is None else x.reshape(NN, 2)
        n = mesh.cell_normal(node=node)
        a = np.sum(n*node[cell[:, 0]])/2.0
        return a


# 产生初始网格
n = 20 
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

if False:
    for i in range(4):
        bc = mesh.entity_barycenter('cell')
        isXEdge = (np.abs(bc[:, 1]) < 1e-8)
        mesh.refine(isxedge)

theta = np.pi/6
model = Model(mesh, theta)
v = model.volume()
val = model()
grad = model.gradient().reshape(-1, 2)
l = np.sqrt(np.sum(grad**2, axis=-1))

# 体积约束
c0 = NonlinearConstraint(model.volume, v, v)

# 点约束
node = model.mesh.entity('node')
nn = model.mesh.number_of_nodes()
iscnode = (np.abs(node[:, 1]) < 1e-8) 
n = iscnode.sum()
idx, = np.nonzero(iscnode)

val = np.ones(n)
i = 2*idx
j = 2*idx
A = csr_matrix((val, (i, j)), shape=(nn*2, nn*2), dtype=np.float)
c1 = LinearConstraint(A, np.zeros(2*nn), np.zeros(2*nn))

options = {'sparse_jacobian': True}

re = minimize(model, node.reshape(-1), 
        jac=model.gradient, method='trust-constr', 
        constraints=[c0, c1], tol=1e-12, options=options)
print(re.x)
print(re.success)
print(re.message)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
axes.quiver(node[:, 0], node[:, 1], grad[:, 0], grad[:, 1])
plt.show()

if False:
    #node = re.x.reshape(NN, 2) # 优化后的节点 
    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    #mesh.find_node(axes, node=node)
    axes.quiver(node[:, 0], node[:, 1], grad[:, 0], grad[:, 1])
    plt.show()

