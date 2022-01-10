# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 20:34:02 2020

@author: mac
"""

import numpy as np

class Model:
    def __init__(self, mesh, theta):
        self.mesh = mesh
        self.theta = theta

    def __call__(self,x,returngrad=True):
        
        def energy(x):       
            x = x.reshape(-1,2)
            NN = x.shape[0]
            y1 = (x[1:NN,1]-x[0:NN-1,1])
            x1 = (x[1:NN,0]-x[0:NN-1,0])
            result = np.sum(((y1**2+x1**2)**0.5)) - np.cos(self.theta)*np.sum((x[NN-1,0] - x[0,0]))
            return result
         
        x = x.reshape(-1,2)
        NN = x.shape[0]
        x_1=x[0:NN,0]
        y_1=np.concatenate([x[1:NN,1],x[0:1,1]])
        y_2=x[0:NN,1]
        x_2=np.concatenate([x[1:NN,0],x[0:1,0]])
        area = np.abs(np.sum(x_1*y_1 - x_2*y_2))*0.5
        
        def diff(x,i):
            y = energy(x)
            x[i] = x[i] + 0.001
            y_new = energy(x)
            return (y_new - y)/0.001
        grad = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            grad[i] = diff(x,i)

        return energy(x),area,grad
           
    def updata(x):
        mesh.node = x.reshape(-1,2)