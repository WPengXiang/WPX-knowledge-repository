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
    tem= x**3 - y/x
    return tem

def real(x):
    tem = 0.2*x**4+1/(5*x)
    return tem 
    
h = 0.1  #步长
x=np.arange(1,3.1,0.1) #x的取值
n = len(x)-1 #迭代次数
r = real(x) #真解
y0 = 2/5

def TwoKutta(F=F, n=n , x=x, h=h, y0 =y0):
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(1, n+1):
        K1 = F(x[i-1],y[i-1])
        K2 = F(x[i-1]+h/2,y[i-1]+h/2*K1)
        y[i] = y[i-1] + h*K2
    return y

def ThreeKutta(F=F, n=n , x=x, h=h, y0 =y0):
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(1, n+1):
        K1 = F(x[i-1],y[i-1])
        K2 = F(x[i-1] + h/2 , y[i-1] + K1*h/2)
        K3 = F(x[i-1]+h , y[i-1]-h*K1+2*h*K2)
        y[i] = y[i-1] + h*(K1+4*K2+K3)/6
    return y

def ThreeHeun(F=F, n=n , x=x, h=h, y0 =y0):
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(1, n+1):
        K1 = F(x[i-1],y[i-1])
        K2 = F(x[i-1] + h/3 , y[i-1] + K1*h/3)
        K3 = F(x[i-1] + h*2/3 , y[i-1] + 2/3*h*K2)
        y[i] = y[i-1] + h*(K1+3*K3)/4
    return y

def FourKutta(F=F, n=n , x=x, h=h, y0 =y0):
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(1, n+1):
        K1 = F(x[i-1],y[i-1])
        K2 = F(x[i-1] + h/3, y[i-1] + K1*h/3)
        K3 = F(x[i-1] + 2*h/3 , y[i-1] - 1/3*h*K1 + h*K2)
        K4 = F(x[i-1] + h , y[i-1] + h*K1 - h*K2 + h*K3)
        y[i] = y[i-1] + h*(K1+3*K2+3*K3+K4)/8
    return y

def FourGill(F=F, n=n , x=x, h=h, y0 =y0):
    y = np.zeros(n+1)
    y[0] = y0
    for i in range(1, n+1):
        K1 = F(x[i-1],y[i-1])
        K2 = F(x[i-1] + h/2 , y[i-1] + K1*h/2)
        K3 = F(x[i-1] + h/2 , y[i-1] + (np.sqrt(2)-1)/2*h*K1 + (1-np.sqrt(2)/2)*h*K2)
        K4 = F(x[i-1] + h , y[i-1] - np.sqrt(2)/2*h*K2 + (1+np.sqrt(2)/2)*h*K3)
        y[i] = y[i-1] + h*(K1+(2-np.sqrt(2))*K2+(2+np.sqrt(2))*K3+K4)/6
    return y


TK = ThreeKutta()
TH = ThreeHeun()
FK = FourKutta()
FG = FourGill()

print("TwoKutta",max(np.abs(r-TwoKutta())))
print("ThreeKutta",max(np.abs(r-TK)))
print("ThreeHeun",max(np.abs(r-TH)))
print("FourKutta",max(np.abs(r-FK)))
print("FourGill",max(np.abs(r-FG)))
