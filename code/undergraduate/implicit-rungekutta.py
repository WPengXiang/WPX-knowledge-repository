#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: runge-kutta.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2020年08月15日 星期六 19时16分02秒
'''  
import numpy as np
import pdb

def F(x,y):#定义函数
    tem= y
    return tem
    
h = 0.1  #步长
x=np.arange(0,1,h) #x的取值
n = len(x)-1  #迭代次数
y0 = 1
def ImplicitKutta(F=F, n=n , x=x, h=h, y0 =y0):
    y = np.zeros(n+1)
    y[0] = y0
    K = 1 
    for i in range(1, n+1):
        K = F(x[i-1]+h/2 , y[i-1]+h*K/2)
        y[i] = y[i-1] + h*K        
    return y

IK= ImplicitKutta()
real = np.exp(x)
#print(IK)
#print(real)
print(max(abs(IK - real)))
