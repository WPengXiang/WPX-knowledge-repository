import pickle
import matplotlib.pyplot as plt
import numpy as np

names = locals()
for i in [32,64,128,256,512,1024]:
    pickle_file = open('diff'+str(i)+'.pkl','rb')
    names['diff' + str(i)] = pickle.load(pickle_file)

n = len(diff32)

xaxis = 0.01*np.arange(n)
for i in [32,64,128,256,512,1024]:
    plt.plot(xaxis,names.get('diff'+str(i)),label = 'ns = '+str(i))
plt.legend()
plt.show()
