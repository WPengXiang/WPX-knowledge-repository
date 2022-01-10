#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: model.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年04月19日 星期一 20时50分41秒
'''  
import numpy as np
from possion_1d import FE_1D_Possion
def fun_c(x):
    return np.exp(x)

def f(x):
    val = -np.exp(x)*(np.cos(x)-2*np.sin(x)-x*np.cos(x)-x*np.sin(x))
    return val


domain = [0,1]
N=4
model = FE_1D_Possion(N,domain,101,8)
A = model.assemble_A(fun_c)
b = model.assemble_b(f)
A,b = model.boundary(A,b,1)
u = np.linalg.solve(A,b)
p,T = model.PbTb()
def solution(x):
    return np.cos(x)*x
print(np.max(np.abs(u-solution(p))))




