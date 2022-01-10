# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 21:49:44 2020

@author: mac
"""

# 创建numpy数组的方法
import numpy as np

a1 = np.zeros(10,dtype = int)
a2 = np.ones((3,5),dtype = float)     
a3 = np.full((3,5),3.14)              #指定形状和值得满矩阵 
a4 = np.arange(0,20,2)                #（开始，结束，步长）
a5 = np.linspace(0,1,5)               #（起始点，结束点，剖分成几个点（包含头尾））
a6 = np.random.random((3,3))          #随机数组
a7 = np.random.normal(0,1,(3,3))      #指定均值和误差的正态分布随机数组
a8 = np.random.randint(0,10,(3,3))    #区间内随机整形数组
a9 = np.eye(3)       #n阶单位矩阵
a10 = np.empty(3)    #未初始化的数组，值为内存空间中任意值

print()
#数据类型大体可分为int uint float complex几种，每种中还可以指定指定字节数

#对数组的操作
#数组的属性
np.random.seed(0)#设置随机种子
x1 = np.random.randint(10,size=6) #一位数组
x2 = np.random.randint(10,size = (3,4)) #二维数组
x3 = np.random.randint(10,size = (3,4,5),dtype = 'int64')#三维数组
print(x3.ndim,x3.shape,x3.size,x3.itemsize,x3.nbytes)
#(维数，每个维度的大小，数组总大小，每个素组元素大小，素组总字节大小)

#数组索引
#一维
print(x1)
print(x1[0],x1[4],x1[-1],x1[-2])#支持负值索引,-1代表倒数第一个，0代表第一个
#多维
print(x2)
print(x2[0,0],x2[1,0],x2[2,-3])
#numpy数组是固定类型的,浮点型会被截断
x1[0]=3.1415
print(x1[0])

#数组切片 x[start:stop:step]默认[0,维度大小，1](不包括stop)
x = np.arange(10)
print(x)
x[::2]
x[::-1]
x[5::-2]
#多维
print(x2)
x2[::-1,::-1]
x2[0]#可以省略空的切片，等于x2[0,:]
x2_sub = x2[1:,2:]
x2_sub[0,0] = 99
x2   #！！！原始数据也会被更改，要想不被更改要用copy（）
x2_sub_copy = x2[1:,2:].copy()
x2_sub_copy[1,1] = 42
x2

#数组的变形
grid = np.arange(1,5).reshape((2,2)) #原始数组的大小必须和变形后数组的大小一致
x = np.array([1,2,3])
x[np.newaxis,:]#np.newaxis的作用就是在这一位置增加一个一维，这一位置指的是np.newaxis所在的位置
#数组的拼接
x = np.array([1,2,3])
y = np.array([3,2,1])
z = np.array([112,12,3])
np.concatenate([x,y,z])
 #也可用于二维数组拼接
grid = np.array([[1,2,3],
                [4,5,6]])
np.concatenate([grid,grid],1)
np.vstack([x,grid]) #vstake(垂直栈)，np.hatack(水平栈)
#数组的分裂
x =[1,2,3,99,99,3,2,1]
x1,x2,x3 = np.split(x,[3,5])#[]里是分裂点位置
grid = np.arange(60).reshape(3,4,5)
print(grid)
grid1,grid2,grid3 = np.vsplit(grid,[1,2])
grid3
