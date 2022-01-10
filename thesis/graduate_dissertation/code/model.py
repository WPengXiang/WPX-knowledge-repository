import numpy as np
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

