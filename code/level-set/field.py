#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: pp.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年11月16日 星期二 14时01分58秒
'''  
import numpy as np
import matplotlib.pyplot as plt
a = 0.146
b=0.292
r=0.04
e = 0.0005
def f1(x,y):
    val = -np.tanh((y-b)/(np.sqrt(2)*e))
    return val

def f2(x,y):
    val = -np.tanh((x-a)/(np.sqrt(2)*e))
    return val

def f3(x,y):
    val = np.tanh((r-np.sqrt((x-(a-r))**2+(y-(b-r))**2))/(np.sqrt(2)*e))
    return val



x1 = np.arange(0,a-r,0.001)
y1 = np.arange(b-r,0.438,0.001)
x2 = np.arange(a-r,0.584,0.001)
y2 = np.arange(0,b-r,0.001)
x3 = np.arange(a-r,0.584,0.001)
y3 = np.arange(b-r,0.438,0.001)
x4 = np.arange(0,a-r,0.001)
y4 = np.arange(0,b-r,0.001)
x1,y1 = np.meshgrid(x1,y1)
x2,y2 = np.meshgrid(x2,y2)
x3,y3 = np.meshgrid(x3,y3)
plt.contour(x1,y1,f1(x1,y1),0)
plt.contour(x2,y2,f2(x2,y2),0)
plt.contour(x3,y3,f3(x3,y3),0)
plt.show()
