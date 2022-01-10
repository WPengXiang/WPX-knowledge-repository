#!/usr/bin/python3

'''!
@file test.py
@author chenchunyu
@date 10/03/2021
@brief 一个测试文件
'''

import numpy as np

class Test:
    '''!
    @brief 测试类
    @todo 完善
    '''
    def __init__(self, a, b):
        '''!
        @brief 初始化函数, 构造对象时调用的函数.
        @param a 第一个参数
        @param b 第二个参数
        '''
        self.a = a
        self.b = b

    def add(self, k, b):
        '''!
        @brief 一个测试函数, 计算 @f[kx+by@f]
        @param k 第一个参数
        @param b 第一个参数
        @return 返回 k*self.a + b*self.b
        '''
        return k*self.a + b*self.b
