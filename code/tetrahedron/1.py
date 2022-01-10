#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: 1.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2020年08月08日 星期六 17时26分06秒
'''  
from fealpy.mesh import MeshFactory
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pdb

mf = MeshFactory()

mesh = mf.boxmesh3d([0, 1, 0, 1, 0, 1], nx = 2, ny = 2, nz = 2, meshtype = 'tet')
node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')
fig = plt.figure()
axes = fig.gca(projection = '3d')
pdb.set_trace()
mesh.add_plot(axes)
mesh.find_node(axes, node=node, showindex=True, fontsize=28)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()


