import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#mpl.use('qt4agg')

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

j = jeu(a,b,g)
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
	llist.append(l)
	alist = []
	alist.append(a)
	flist = []
	flist.append(cout)
	while (k<400 and d >= dmin) :	
		coutav = cout
		grad = gradf1(x,y,gradg, a)
		hess = gaussf1(x, gradg, a)
		hlm = hess*(1+l)
		d = - grad/hlm
		cout = f(x,y,a-d)
		k = k+1
		if(cout < coutav):
			a += d
			l = l/10
			
		else:
			l *=10
			
		dlist.append(d)
		flist.append(cout)
		alist.append(a)
		llist.append(l)
	
	'''
	plt.plot(range(0,k), dlist)
	plt.show()
	plt.plot(range(0,k+1), alist)
	plt.show()
	plt.plot(range(0,k+1), llist)
	plt.show()
	plt.plot(range(0,k+1), flist)
	plt.show()'''
	print "convergence au bout de ", k, " iterations"
	return a

def plotk():
	#changer ce que retourne leven : mettre k
	jeux = [jeu(a,b ,g) for b in range(1,60)]
	ks = [leven(f, gradf1, gradg, gaussf1, X, j, 0.0001) for j in jeux]
	plt.plot(range(1,60), ks)
	plt.show()

def plotb():
	#changer ce que retourne leven : mettre a
	jeux = [jeu(a,b*0.1 ,g) for b in range(1,8)]
	aa = [leven(f, gradf1, gradg, gaussf1, X, j, 0.0001) for j in jeux]
	for i in range(0,len(jeux)-1):
		plt.plot(X,jeux[i], '+')
		plt.plot(X, g(X,aa[i]))
		plt.show()

#leven(f, gradf1, gradg, gaussf1, X, j, 0.001)

plotb()
