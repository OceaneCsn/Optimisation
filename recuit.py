import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mpl.use('qt4agg')

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import math

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
    
def g(x,y):
	return f(x) + f(y)

fig = plt.figure() #opens a figure environment
ax = fig.gca(projection='3d') #to perform a 3D plot
X = np.arange(-5, 5, 0.25) #x range
Y = np.arange(-5, 5, 0.25) #y range
X, Y = np.meshgrid(X, Y)
Z = X**4 -X**3 - 20*X**2 + X + 1 + Y**4 - Y**3 - 20*Y**2 + Y + 1
my_col = cm.jet(Z/np.amax(Z))
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors = my_col,linewidth=0, antialiased=False) #plot definition and options
#plt.show()

x = np.arange(-5, 5, 0.25)
'''plt.plot(x, [f(X) for X in x])
plt.plot()
plt.show()
'''
print("minimum : ",x[np.argmin(f(x))])

def recuit(f,x0, kappa, kappa2, tmax):
	x = x0
	t = 1
	xlist = []
	while(t<tmax):
		T = 1/t 
		sol = x + np.random.normal(0,math.sqrt(kappa*np.exp(-1/(1000*T))))
		if (f(sol) < f(x)):
			x = sol
		else:
			if(rd.random()<kappa2*np.exp(-1/1000*T)):
				x = sol
		t +=1
		xlist.append(x)

	print(x,f(x))
	return xlist, x

def histogramme(tmax):
	hi = []
	for i in range(0,tmax):
		hi.append(recuit(f,-1,10,0.5,10000)[1])
	plt.hist(hi)
	plt.show()
	
	
def traj():
	res = recuit(f,-1,10,0.5,10000)

	plt.plot(x, [f(X) for X in x])
	plt.plot(res, [f(xs) for xs in res], color = "red")
	plt.xlim([-5,5])
	plt.ylim([-150,300])
	plt.plot()
	plt.show()
	
def recuit2(f,X0, kappa, kappa2, tmax):
	x = X0[0]
	y = X0[1]
	t = 1
	xlist = []
	ylist = []
	zlist = []
	while(t<tmax):
		T = 1/t 
		D = np.random.normal(0,math.sqrt(kappa*np.exp(-1/(1000*T))), size = 2)
		solx, soly = x +D[0], y+D[1]
		if (g(solx,soly) < f(x,y)):
			x,y = solx, soly
		else:
			if(rd.random()<kappa2*np.exp(-1/1000*T)):
				x,y = solx, soly
		t +=1
		xlist.append(x)
		ylist.append(y)
		zlist.append(f(x,y))
		
	print(x,y,g(x,y))
	return [xlist,ylist,zlist],[x,y,g(x,y)]

xs, ys, zs = recuit2(g,[-1, -3, -1],10,0.5,10000)[0]
ax.plot(xs[5000:], ys[5000:], zs[5000:], color = "orange")
ax.set_zlim(-200, 300)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
#plt.xlim([-5,5])
#plt.ylim([-150,300])
plt.show()


def histogramme2(tmax):
	hix = []
	hiy = []
	for i in range(0,tmax):
		r = recuit2(g,[-1, -3, -1],10,0.5,8000)[1]
		hix.append(r[0])
		hiy.append(r[1])
	plt.hist2d(hix, hiy)
	plt.show()
	
def sismo():
	r = recuit2(g,[-1, -3, -1],10,0.5,8000)[0]
	Xres = r[0]
	Yres = r[1]
	plt.plot(range(len(Xres)),Xres)
	plt.plot(range(len(Yres)),Yres)
	plt.show()

#histogramme2(100)
sismo()
