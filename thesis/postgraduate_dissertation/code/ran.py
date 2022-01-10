import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from fealpy.mesh import IntervalMesh
from model import Model

# 产生初始网格
n = 20
np.random.seed(1000)
x = np.linspace(0, 20, n)
y = np.random.random_integers(0,20,n)
y[0]=0
y[-1]=0
node = np.zeros((n, 2), dtype=np.float)
node[:, 0] = x
node[:, 1] = y
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
mesh.add_plot(axes,showaxis=True,nodecolor='white')

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
               options={'maxiter': 2000, 'ftol': 1e-06})

x0 = model.x0.reshape(-1)
a0 = np.degrees(np.arctan((x0[3]-x0[1])/(x0[2]-x0[0])))
af = np.degrees(np.arctan((re.x[3]-re.x[1])/(re.x[2]-re.x[0])))
print("优化模型为random")
print("平衡时接触角为",np.degrees(theta))
print("初始能量函数值",val)
print("优化后能量函数值",re.fun)
print("迭代次数",re.nit)
print("结点个数",n)
print("初始接触角为",a0)
print("最终接触角",af)
print("误差为",np.abs(af-np.degrees(theta)))
print("\n")
plt.subplot(2,1,2)
node = re.x.reshape(-1, 2) # 优化后的节点
grad = re.jac.reshape(-1,2)
axes = fig.gca()
mesh.add_plot(axes,showaxis=True,nodecolor='white')
plt.show()
