#Imports from the matplotlib library
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mpl.use('qt4agg')

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
#--------------------------------------

def g(x,a):
  return (x**a[0])*np.exp(-x*a[1])
  
def f(x, y, a):
  return 0.5*sum((y-g(x, a))**2)
  
def data(a,b):
  x = pl.frange(0,3,0.01)[1:]
  y = g(x, a)+b*np.random.randn(len(x))
  return x, y

a = (2.0,3.0)
b = 0.01
xx, yy = data(a, b)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.arange(1.5, 2.5, 0.05)
y = np.arange(2.5, 3.5, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([f(xx,yy,[x,y]) for x,y in zip(np.ravel(X), np.ravel(Y))])


Z = zs.reshape(X.shape)
my_col = cm.jet(Z/np.amax(Z))
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = my_col, antialiased=False)

ax.set_xlabel('a1')
ax.set_ylabel('a2')
ax.set_zlabel('f')

plt.show()

'''#Definition of what to plot
fig = plt.figure() #opens a figure environment
ax = fig.gca(projection='3d') #to perform a 3D plot
X = np.arange(-2, 4, 0.25) #x range
Y = np.arange(-2, 4, 0.25) #y range
X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
#Z=(X-Y)**4+2*X**2+Y**2-X+2*Y #defines the function values
Z = 0.5*sum((y-(x**X)*np.exp(-x*Y))**2)

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options

#Runs the plot command
plt.show()
'''
