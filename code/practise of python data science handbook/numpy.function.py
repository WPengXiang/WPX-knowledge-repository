# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 22:58:19 2020

@author: mac
"""

import numpy as np
np.random.seed(0)
#!!!numpy尽量不用循环
def compute_reciprocals(value):
    output = np.empty(len(value))
    for i in range(len(value)):
        output[i] = 1.0/value[i]
    return output
value = np.random.randint(1,10,size=5)
compute_reciprocals(value)

big_array = np.random.randint(1,100,size=1000000)
%timeit compute_reciprocals(big_array)
%timeit 1.0/big_array

#数组间运算就是对应元素的运算
np.arange(5)/np.arange(1,6)
x = np.arange(9).reshape(3,3)
2**x

#函数与符号对应关系
#(np.add ,+)(np.subtract,-)(np.negative,-复数运算)（np.multiply,*）
#（np.divide,/）（np.power,**）(np.mod,%)
#(np,floor_divide，//)地板除法：//除法只取结果的整数部分

x = np.arange(-2,3)
#绝对值或者np.absolute
np.abs(x)  
x = np.array([3 - 4j,4 - 3j,2 + 0j,0 + 1j])#也可以处理负数,返回幅度
np.abs(x)
#三角函数
theta = np.linspace(0,np.pi,3)
np.sin(theta)#np.sin,np.cos,np.tan,np.arcsin,np.arccos,np.arctan
#指数
x = [1,2,3]
print("e^x =",np.exp(x))
print("4^x = ",np.power(4,x))
#指数
x = [1,2,4,10]
print("ln(x) = ",np.log(x))
print("log2(x) = ",np.log2(x))#还有np.log10

