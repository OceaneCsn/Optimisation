#! /usr/bin/env python
# -*- coding:utf-8 -*-

#Imports from the matplotlib library
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#mpl.use('qt4agg')

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
#--------------------------------------

#Definition of what to plot
'''fig = plt.figure() #opens a figure environment
ax = fig.gca(projection='3d') #to perform a 3D plot
X = np.arange(-2, 2, 0.25) #x range
Y = np.arange(-2, 2, 0.25) #y range
X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)
Z=(X-Y)**4+2*X**2+Y**2-X+2*Y #defines the function values
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=False) #plot definition and options
'''
#Runs the plot command
#plt.show()

#fonction à optimiser
def f(X,Y):
	return (X-Y)**4+2*X**2+Y**2-X+2*Y

#gradient de la fonction à optimiser
def g(x,y):
	return [4*(x-y)**3+4*x-1, -4*(x-y)**3 +2*(y+1)]

#méthode de descente
def descente(f,g, alpha, X0):
	x = X0[0]
	y = X0[1]
	k = 0
	dlist = []
	while k < 100:	
		d = -g(x,y)[0]
		dlist.append(d)
		x += alpha*d
		y += alpha*d
		k = k+1
	plt.xlabel('Iterations')
	plt.ylabel('d')
	plt.title("Valeurs de d au cours du temps")
	plt.plot(range(0,100),dlist)
	plt.show()
	return x,y
	
print descente(f,g,0.09, [1,1])
