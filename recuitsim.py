import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mpl.use('qt4agg')

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import random as rd

'''fig = plt.figure() #opens a figure environment
ax = fig.gca(projection='3d') #to perform a 3D plot
X = np.arange(-5, 5, 0.25) #x range
Y = np.arange(-5, 5, 0.25) #y range
X, Y = np.meshgrid(X, Y)
Z = X**4 -X**3 - 20*X**2 + X + 1
my_col = cm.jet(Z/np.amax(Z))
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = my_col,linewidth=0, antialiased=False) #plot definition and options
plt.show()
'''

def f(X):
    return X**4 -X**3 - 20*X**2 + X + 1

x = np.arange(-5, 5, 0.25)
'''plt.plot(x, [f(X) for X in x])
plt.plot()
plt.show()'''

print("minimum : ",x[np.argmin(f(x))], np.argmin(f(x)))

def recuit(f,x0, kappa, kappa2, kmax):
    x = x0
    t = 0
    while(t < tmax):
	T = 1/t
	sol = x + np.random.normal(0,sqrt(kappa*np.exp(-1/1000*T)
	if(f(sol) < f(x)):
	    x += np.random.normal(0,sqrt(kappa*np.exp(-1/1000*T)
	else:
	    if(rd.random()<kappa2*np.exp(-1/1000*T)):
		x += np.random.normal(0,sqrt(kappa*np.exp(-1/1000*T)
	t +=1

    print(x,f(x))

recuit(f,-1,10,0.5,10000)

