import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint, LinearConstraint
from scipy.optimize import minimize
from fealpy.mesh import IntervalMesh
from model import Model

# 产生初始网格
for i in (0,1):
    n = 40
    x1 = np.linspace(0, np.pi, n)
    y1 = np.sin(x1)
    x2 = np.linspace(-np.power(3/2,1/3), np.power(3/2,1/3), n)
    y2 = -x2**2+np.power(3/2,2/3)
    x = [x1,x2]
    y = [y1,y2]
    node = np.zeros((n, 2), dtype=np.float)
    node[:, 0] = x[i]
    node[:, 1] = y[i]
    cell = np.zeros((n, 2), dtype=np.int)
    cell[:, 0] = range(n-1, -1, -1)
    cell[:n-1, 1] = range(n-2, -1, -1)
    cell[-1, 1] = n-1
    mesh = IntervalMesh(node, cell)

#建立模型
    theta = np.pi/6
    model = Model(mesh, theta)
    v = model.volume()
    NN = model.mesh.number_of_nodes() 
    val = model()
    grad = model.gradient().reshape(-1, 2)
    fig = plt.figure()
    plt.subplot(2,1,1)
    node = model.x0
    axes = fig.gca()
    mesh.add_plot(axes,nodecolor='white',showaxis=True)

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
                   options={'maxiter': 1000, 'ftol': 1e-08})
    x0 = model.x0.reshape(-1)
    letter = ['sinx', 'x^2+(3/2)^(2/3)']
    print("优化模型为",letter[i])
    print("平衡时接触角度为",np.degrees(theta))
    print("面积为",v)
    print("结点个数",n)
    print("初始能量函数值",val)
    print("优化后能量函数值",re.fun)
    print("迭代次数",re.nit)
    print("初始接触角",np.degrees(np.arctan((x0[3]-x0[1])/(x0[2]-x0[0]))))
    print("优化后接触角",np.degrees(np.arctan((re.x[3]-re.x[1])/(re.x[2]-re.x[0]))))
    print("\n")
    plt.subplot(2,1,2)
    node = re.x.reshape(-1, 2) # 优化后的节点
    grad = re.jac.reshape(-1,2)
    axes = fig.gca()
    mesh.add_plot(axes,nodecolor='white',showaxis=True)
plt.show()
