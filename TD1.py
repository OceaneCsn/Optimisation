#! /usr/bin/env python
# -*- coding:utf-8 -*-

#Imports from the matplotlib library
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mpl.use('qt4agg')

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
#--------------------------------------

#Definition of what to plot
fig = plt.figure() #opens a figure environment
ax = fig.gca(projection='3d') #to perform a 3D plot
X = np.arange(-5, 5, 0.25) #x range
Y = np.arange(-5, 5, 0.25) #y range
X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
#Z=(X-Y)**4+2*X**2+Y**2-X+2*Y #defines the function values
#Z = X**2-Y**2
Z = X**4 -X**3 - 20*X**2 + X + 1 + Y**4 - Y**3 - 20*Y**2 + Y + 1
my_col = cm.jet(Z/np.amax(Z))
#my_col = cm.jet(Z)
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = my_col,linewidth=0, antialiased=False) #plot definition and options

#Runs the plot command
#plt.show()

X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25) 
X, Y = np.meshgrid(X, Y)

#fonction à optimiser
def f(X,Y):
	return (X-Y)**4+2*X**2+Y**2-X+2*Y

def f2(x,y):
	return x**2-y**2
	
def f3(X,Y):
	return X**4 -X**3 - 20*X**2 + X + 1 + Y**4 - Y**3 - 20*Y**2 + Y + 1
	
#print np.min(f3(X,Y))
#print np.argmin(f3(X,Y))
#print f(X,Y)[6][8]
#print np.arange(-2, 2, 0.25)[6], np.arange(-2, 2, 0.25)[8]

#gradient de la fonction à optimiser
def g(x,y):
	return [4*(x-y)**3+4*x-1, -4*(x-y)**3 +2*(y+1)]
	
def g3(x,y):
	return [4*x**3-3*x**2-40*x+1, 4*y**3-3*y**2-40*y+1]
	
def g2(x,y):
	return [2*x, -2*y]

#méthode de descente
def descente(f,g, alpha, X0):
	x = X0[0]
	y = X0[1]
	z = f(x,y)
	k = 0
	dlist = []
	d = g(x,y)
	xlist = []
	ylist = []
	zlist = []
	xlist.append(x)
	ylist.append(y)
	zlist.append(z)
	while k<20:	
		d = g(x,y)
		dlist.append(np.linalg.norm(d))
		x -= alpha*d[0]
		y -= alpha*d[1]
		z = f(x,y)
		xlist.append(x)
		ylist.append(y)
		zlist.append(z)
		k = k+1
	#plt.xlabel('Iterations')
	#plt.ylabel('d')
	#plt.title("Valeurs de d au cours du temps")
	#plt.plot(range(0,k),dlist)
	#plt.show()
	#print "Descente en ",k,"itérations.\nCoordonnées de l'optimum : ", x,y,z, "\nDerniere valeur de la norme de d :",np.linalg.norm(d)
	print(np.linalg.norm(d),k,z)		
	return xlist, ylist, zlist
	

def h(x,y):
	return np.matrix([[12*(x-y)**2 +4 , -12*(x-y)**2],[-12*(x-y)**2 ,12*(x-y)**2 +2 ]])
	

def h2(x,y):
	return np.matrix([[2 ,0],[0,-2]])
	

def h3(x,y):
	return np.matrix([[12*x**2-6*x-40 ,0],[0,12*y**2-6*y-40]])
	
	
'''x1, y1, z1 = descente(f3,g3,0.01, [-0.5,-0.5])
x2, y2, z2 = descente(f3,g3,0.01, [-0.5,0.5])
x3, y3, z3 = descente(f3,g3,0.01, [0.5,0.5])
x4, y4, z4 = descente(f3,g3,0.009, [0.5,-0.5])

ax.plot(x1, y1, z1, color = "orange")
ax.plot(x2, y2, z2, color = "r")
ax.plot(x3, y3, z3, color = "yellow")
ax.plot(x4, y4, z4, color = "green")
plt.show()'''



def descenteNewton(f,g, h, alpha, X0):
	x = X0[0]
	y = X0[1]
	z = f(x,y)
	k = 0
	dlist = []
	d = np.dot(g(x,y),np.linalg.inv(h(x,y)))
	xlist = []
	ylist = []
	zlist = []
	xlist.append(x)
	ylist.append(y)
	zlist.append(z)
	while np.linalg.norm(d)>0.0001:	
		d = np.dot(g(x,y),np.linalg.inv(h(x,y)))
		d = d.tolist()[0]
		dlist.append(np.linalg.norm(d))
		x -= d[0]
		y -= d[1]
		z = f(x,y)
		xlist.append(x)
		ylist.append(y)
		zlist.append(z)
		k = k+1
	#print(z)
	print(np.linalg.norm(d),k, z)		
	return xlist, ylist, zlist
		
x1, y1, z1 = descenteNewton(f3,g3,h3,0.009, [0.5,0.5])
x2, y2, z2 = descenteNewton(f3,g3,h3,0.009, [1,-2])
x3, y3, z3 = descenteNewton(f3,g3,h3,0.009, [-1,-1])

#x3, y3, z3 = descenteNewton(f,g,h,0.009, [1,1])
x4, y4, z4 = descenteNewton(f3,g3,h3,0.009, [0,-0.5])

ax.plot(x1, y1, z1, color = "orange")
ax.plot(x2, y2, z2, color = "r")
ax.plot(x3, y3, z3, color = "yellow")
ax.plot(x4, y4, z4, color = "green")
plt.show()
