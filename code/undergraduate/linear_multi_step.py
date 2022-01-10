import numpy as np
import matplotlib.pyplot as plt

class linear_multi_step():

    def __init__(self,x0,y0,x,h):
        '''
        x0:初值x
        y0:初值y
        x :终值x
        h :迭代步长

        dy/dx = y
        '''
        self.x0 = x0
        self.y0 = y0
        self.x = x
        self.h = h
        self.n = int((self.x-self.x0)/h)

    def implicit_Adams(self):
        xs = np.linspace(self.x0,self.x,self.n+1)
        ys = np.zeros(len(xs))
        ys[0] = self.y0
        for i in range(self.n):
            ys[i+1] = ((1-h/(2*xs[i]))*ys[i]+h/2*(xs[i+1]**3+xs[i]**3))/(1+h/(2*xs[i+1]))
        return xs,ys
    
    def Improved_Euler_formula(self):
        xs = np.linspace(self.x0,self.x,self.n+1)
        ys = np.zeros(len(xs))
        ys[0] = self.y0
        for i in range(self.n):
            y = ys[i] +h*(xs[i]**3-ys[i]/xs[i])
            ys[i+1] = ys[i] +h/2*(xs[i]**3-ys[i]/xs[i] + xs[i+1]**3-y/xs[i+1])
        return xs,ys

    def test(self,method = 'implicit_Adams',plot = True):
        #y = 1/5*x**4 +1/(5*x) 
        
        if method == 'implicit_Adams':
            xs,ys = self.implicit_Adams()
            ss = self.exact_solution(xs)
            error = abs(ss - ys)
            print("max error {:.5f}".format(max(error)))
            print(error)

        
        if method == 'Improved_Euler_formula':
            xs,ys = self.Improved_Euler_formula()
            ss = self.exact_solution(xs)
            error = abs(ss - ys)
            print("max error {:.5f}".format(max(error)))
            print(error)

        if plot:
            plt.plot(xs,ys,label = method)
            plt.legend()
            plt.show()
            plt.plot(xs, ss, label = 'exact')
            plt.legend()
            plt.show()
    
    def exact_solution(self,x):
        s = 1/5*x**4 + 1/(5*x)
        return s

x0=1
y0=0.4
x =2
h = 0.1

Test = linear_multi_step(x0,y0,x,h)
Test.test(method = 'implicit_Adams')
#Test.test(method = 'Improved_Euler_formula')
