import pickle
import matplotlib.pyplot as plt
import numpy as np

pickle_file1 = open('diff64-1no.pkl','rb')
no = pickle.load(pickle_file1)

pickle_file2 = open('diff64-1.pkl','rb')
yes = pickle.load(pickle_file2)

n = len(no)

xaxis = 0.01*np.arange(n)

plt.plot(xaxis,no,label = 'no reinitalization')
plt.plot(xaxis,yes,label = 'reinitialization')
plt.legend()
plt.title('NS = 64')
plt.show()
