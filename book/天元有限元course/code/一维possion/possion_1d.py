#!/usr/bin/python3
'''    	
	> Author: wpx
	> File Name: mdoel.py
	> Author: wpx
	> Mail: wpx15673207315@gmail.com 
	> Created Time: 2021年04月17日 星期六 15时50分16秒
'''  
import sys
import numpy as np
sys.path.append(r'~/wpx-repository/wangpengxiang-knowledge/book/天元有限元course/code/一维possion/GaussLegendreQuadrature.py')
from GaussLegendreQuadrature import GaussLegendreQuadrature
from basicfuntion import basicfuntion
class FE_1D_Possion:
    '''
    N:网格单元数
    Np:网格点个数
    Nb:有限元全部基函数个数
    Nlb:有限元局部基函数个数
    '''
    def __init__(self,N,domain,funtype,gauss):
        self.N = N
        self.Np = N+1
        self.domain = domain
        self.funtype = funtype
        self.gauss = gauss
        fun = basicfuntion()
        if funtype==101:
            self.trial = fun.liner_1
            self.test = fun.liner_1
        elif funtype==102:
            self.trial = fun.liner_2
            self.test = fun.liner_2
            

    def PT(self):
        N = self.N
        Np = self.Np
        domain = self.domain
        P = np.linspace(domain[0],domain[-1],Np)
        T = np.zeros((2,N),dtype=np.int_)
        T[0] = np.arange(N)
        T[1] = np.arange(1,N+1)
        T = T
        return P,T

    def PbTb(self):
        funtype = self.funtype
        if funtype == 101:
            self.Nb = self.Np
            self.Nlb = 2
            return self.PT()
        elif funtype == 102:
            self.Nb = 2*self.N+1
            self.Nlb = 3
            Nb = self.Nb
            Nlb =self.Nlb
            N = self.N
            domain = self.domain
            Pb = np.linspace(domain[0],domain[-1],Nb)
            Tb = np.zeros((Nlb,N),dtype=np.int_)
            Tb[0] = np.arange(0,Nb-2,2)
            Tb[1] = np.arange(1,Nb-1,2)
            Tb[2] = np.arange(2,Nb,2)
            return Pb,Tb

    def assemble_A(self,c):
        Pb,Tb = self.PbTb()
        Nb=self.Nb
        N = self.N
        self.c = c
        guass = self.gauss
        Nlb = self.Nlb 
        trial = self.trial
        test = self.test
        A = np.zeros((Nb,Nb))
        point,weight = GaussLegendreQuadrature(guass).get_quadrature_points_and_weights()
        point = point[:,1]
        for i in np.arange(N):
            cell = Pb[Tb[(0,-1),i]]
            h = cell[1]-cell[0]
            x = cell[0]+h*point
            for j in np.arange(Nlb):
                for k in np.arange(Nlb):
                     A[Tb[k,i],Tb[j,i]] += h * (c(x)*trial(x,cell,1,j)*test(x,cell,1,k) @ weight)
        return A

    def assemble_b(self,f):
        Pb,Tb = self.PbTb()
        Nb=self.Nb
        N = self.N
        gauss = self.gauss
        Nlb = self.Nlb 
        test = self.test
        b = np.zeros(Nb)
        point,weight = GaussLegendreQuadrature(gauss).get_quadrature_points_and_weights()
        point = point[:,1]
        for i in np.arange(N):
            cell = Pb[Tb[(0,-1),i]]
            h = cell[1]-cell[0]
            x = cell[0]+h*point
            for j in np.arange(Nlb):
                b[Tb[j,i]] += h * (f(x)*test(x,cell,0,j) @ weight)
        return b

    def boundary(self,A,b,boundary_type):
        if boundary_type==0:
            N = self.Nb
            A[0] = np.zeros(N)
            A[-1] = np.zeros(N)
            A[0,0] = 1
            A[-1,-1]=1
            b[0] = 0
            b[-1] = np.cos(1)
        if boundary_type==1:
            N = self.Nb
            test = self.test
            domain = self.domain
            P,T = self.PT()
            bb = (np.cos(1)-np.sin(1))*self.c(domain[1])
            b[-1] = b[-1]+bb
            A[0] = np.zeros(N)
            A[0,0] = 1
            b[0] = 0
        return A,b



