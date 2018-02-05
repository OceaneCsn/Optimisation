import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mpl.use('qt4agg')

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

def g(x,a):
	return np.exp(-a*x)


def gradg(x,a):
	return -x*np.exp(-a*x)
	
	
b = 0.1
a = 2

def jeu(a, b, g):
	res = []
	for x in range(0,301): 
		res.append(g(a,x/100.0)+b*np.random.randn())
	return res

jeu = jeu(a,b,g)
X = np.linspace(0,3,301)

#plt.plot(X,jeu, '+')
#plt.plot(X, g(X,a))
#plt.show()

def f(x,y,a):
	return 0.5*np.sum((y-g(x,a))**2)	
	
def gradf1(x, y, gradg, a):
	return -np.sum((y-g(x,a))*gradg(x,a))
	
#print(gradf1(X, jeu, g, a))

def gaussf1(x,gradg, a):
	return np.sum(gradg(x,a)**2)


def leven(f,gradf1, gradg, gaussf1, x, y, dmin):
	a = 1.5
	l = 0.001
	cout = f(x,y,a)
	k = 0
	d = dmin
	dlist = []
	llist = []
	alist = []
	flist = []
	while (k<200 and d >= dmin) :	
		coutav = cout
		flist.append(cout)
		grad = gradf1(x,y,gradg, a)
		hess = gaussf1(x, gradg, a)
		hlm = hess*(1+l)
		#print(hlm)
		d = - grad/hlm
		cout = f(x,y,a-d)
		k = k+1
		if( cout < coutav):
			a += d
			l = l/10
			
		else:
			l *=10
	print(a)

leven(f, gradf1, gradg, gaussf1, X, jeu, 0.001)
#print(gaussf1(X,gradg,a))
