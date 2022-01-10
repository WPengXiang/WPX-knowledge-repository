#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: basicfuntion.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年04月21日 星期三 13时53分07秒
''' 
import numpy as np
class basicfuntion():
    def __int__(self):
        pass
   
    def liner_1(self,x,cell,derivate,index):
        '''
        derivate几阶导数
        index 第几个局部奇函数
        '''
        h = cell[-1]-cell[0]
        if index == 0:
            if derivate == 0:
                return (cell[1]-x)/h
            elif derivate == 1:
                return -1.0*np.ones(x.shape[0])/h
            else:
                return 0
        elif index == 1:
            if derivate == 0:
                return (x-cell[0])/h
            elif derivate == 1:
                return 1.0*np.ones(x.shape[0])/h
            else:
                return 0
        else:
            print("格式不符合")

    def liner_2(self,x,cell,derivate,index):
        def r0(y):
            return 2*y*y-3*y+1.0
        def r1(y):
            return -4.0*y*y+4*y
        def r2(y):
            return 2.0*y*y-y
        
        h = cell[-1]-cell[0]
        
        if index == 0:
            if derivate == 0:
                return r0((x-cell[0])/h)
            elif derivate == 1:
                return (4*(x-cell[0])/h-3)/h
            elif derivate == 2:
                return 4*np.ones(x.shape[0])
            else:
                return 0

        elif index == 1:
            if derivate == 0:
                return r1((x-cell[0])/h)
            elif derivate == 1:
                return (-8*(x-cell[0])/h+4)/h
            elif derivate == 2:
                return -8.0/h*np.ones(x.shape[0])
            else:
                return 0
            
        elif index == 2:
            if derivate == 0:
                return r2((x-cell[0])/h)
            elif derivate == 1:
                return (4*(x-cell[0])/h-1)/h
            elif derivate == 2:
                return 4*np.ones(x.shape[0])/h
            else:
                return 0
            


   


