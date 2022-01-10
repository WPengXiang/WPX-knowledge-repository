#!/usr/bin/env python3
# 

import sys

import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from fealpy.mesh import MeshFactory
from fealpy.mesh import LagrangeTetrahedronMesh 

class LagrangeTetrahedronMeshTest():
    def __init__(self):
        pass

    def show_mesh(self, p=2, plot=True):

        mf = MeshFactory()

        mesh = mf.boxmesh3d([0, 1, 0, 1, 0, 1], nx =2, ny=2, nz=2, meshtype='tet')
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        cell = mesh.entity('cell')

        #ltmesh = LagrangeTetrahedronMesh(node, cell, p=p)
        #node = ltmesh.entity('node')
        #ltmesh.print()

        if plot:
            fig = plt.figure()
            axes = fig.gca(projection = '3d')
            mesh.add_plot(axes)
            mesh.find_node(axes, node=node, showindex=True, fontsize=28)
            mesh.find_edge(axes, showindex=True)
            mesh.find_cell(axes, showindex=True)
            plt.show()

    def save_mesh(self, p=2, fname='test.vtu'):
        mf = MeshFactory()

        mesh = mf.boxmesh2d([0, 1, 0, 1], nx =2, ny=2, meshtype='tri')
        node = mesh.entity('node')
        cell = mesh.entity('cell')

        mesh = LagrangeTriangleMesh(node, cell, p=p)
        mesh.to_vtk(fname=fname)


test = LagrangeTriangleMeshTest()
test.show_mesh(p=p)
