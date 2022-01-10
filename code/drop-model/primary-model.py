import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import IntervalMesh

class Model:

    def __init__(self, mesh, theta):
        self.mesh = mesh
        self.theta = theta

    def __call__(self, node=None, returngrad=True):
        mesh = self.mesh
        theta = self.theta
        gamma = np.cos(theta)           #接触角
        h = mesh.entity_measure('cell')   #每一个单元的长度
        a = np.sum(h[:-1])                #液气面长度
        b = np.sum(h[-1])                       #固液面的长度
        val = a - gamma*b               #能量函数
        if returngrad == True:
            NN = mesh.number_of_nodes()    #节点数
            v = mesh.cell_tangent()/h[:, np.newaxis]  
            g = np.zeros((NN, 2), dtype=mesh.ftype)  #新建ｎ行两列的，储存切线
            np.add.at(g, (cell[:-1, 1], np.s_[:]), v[:-1])
            np.subtract.at(g, (cell[:-1, 0], np.s_[:]), v[:-1])

            g[cell[-1, 0], :] = g[cell[-1, 0], :] + gamma*v[-1]
            g[cell[-1, 1], :] = g[cell[-1, 1], :] - gamma*v[-1]

            g[cell[-1, 0]] = np.sum(g[cell[-1, 0]]*v[-1])*v[-1]
            g[cell[-1, 1]] = np.sum(g[cell[-1, 1]]*v[-1])*v[-1]
            return val, g
        return val

    def volume(self):        #计算面积
        mesh = self.mesh      
        cell = mesh.entity('cell')
        node = mesh.entity('node')
        n = mesh.cell_normal()    #第一列为纵坐标差，第二列为横坐标差　
        a = np.sum(n*node[cell[:, 0]])/2.0    #node[cell[:,0]]相当于遍历所有点
        return a

# 产生初始网格
n = 50
nn = 2*n-2
x = np.linspace(0, np.pi, n)
y = np.sin(x)
node = np.zeros((nn,2),dtype=np.float)
node[0:n,0] = np.linspace(0,np.pi,n)
node[n:nn,0] = np.linspace(0,np.pi,n)[1:n-1]
node[0:n,1] = np.sin(node[0:n, 0])
cell = np.zeros((nn, 2), dtype=np.int)
cell[:, 0] = range(nn-1, -1, -1)
cell[:nn-1, 1] = range(nn-2, -1, -1)
cell[-1, 1] = nn-1
mesh = IntervalMesh(node, cell)
bc = mesh.entity_barycenter('cell')

#建立模型
theta = np.pi/6
model = Model(mesh, theta)
v = model.volume()
val, grad = model()

#画图
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
axes.quiver(node[:, 0], node[:, 1], grad[:, 0], grad[:, 1])
plt.show()
