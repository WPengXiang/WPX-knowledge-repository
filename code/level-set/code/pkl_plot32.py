import pickle                                                                   
import matplotlib.pyplot as plt                                                 
import numpy as np                                                              
                                                                             
names = locals()                                                                
for i in range(5):                                                              
    pickle_file = open('diff32-'+str(i+1)+'.pkl','rb')                          
    names['diff32_' + str(i+1)] = pickle.load(pickle_file)                      
                                                                             
n = len(diff32_1)                                                               
                                                                             
xaxis = 0.01*np.arange(n)                                                       
for i in range(5):
    plt.plot(xaxis,names.get('diff32_'+str(i+1)),label = 'p = '+str(i+1))       
plt.legend()                                                                    
plt.show()              
