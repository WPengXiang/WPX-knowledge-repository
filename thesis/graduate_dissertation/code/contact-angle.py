import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint, LinearConstraint
from scipy.optimize import minimize
from fealpy.mesh import IntervalMesh
from model import Model

for theta in (np.pi/6,np.pi/3,np.pi/2):
# 产生初始网格 
    n = 40
    x = np.linspace(0, np.pi, n)
    y = np.sin(x)
    node = np.zeros((n, 2), dtype=np.float)
    node[:, 0] = x
    node[:, 1] = y
    cell = np.zeros((n, 2), dtype=np.int)
    cell[:, 0] = range(n-1, -1, -1)
    cell[:n-1, 1] = range(n-2, -1, -1)
    cell[-1, 1] = n-1
    mesh = IntervalMesh(node, cell)


#建立模型
    theta = theta 
    model = Model(mesh, theta)
    v = model.volume()
    val = model()
    grad = model.gradient().reshape(-1, 2)
    NN = mesh.number_of_nodes()
    if theta == np.pi/2 :
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
                   method='SLSQP', jac = model.gradient,
                   constraints=cons1,bounds=bnds,
                   options={'maxiter': 2000, 'ftol': 1e-08})

    print("优化模型为sinx")
    print("初始接触角度为",np.degrees(theta))
    print("结点个数",n)
    print("初始能量函数值",val)
    print("优化后能量函数值",re.fun)
    print("迭代次数",re.nit)
    print("最终接触角",np.degrees(np.arctan((re.x[3]-re.x[1])/(re.x[2]-re.x[0]))))
    print("\n")
plt.subplot(2,1,2)
node = re.x.reshape(-1, 2) # 优化后的节点
grad = re.jac.reshape(-1,2)
axes = fig.gca()
mesh.add_plot(axes)
axes.quiver(node[:, 0], node[:, 1], grad[:, 0], grad[:, 1])
plt.show()
