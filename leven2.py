import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#mpl.use('qt4agg')

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


def g(a,x):
	return x**a[0]*np.exp(-a[1]*x)
	
def grad_g(x,a):
	return [np.log(x)*x**a[0]*np.exp(-a[1]*x), -x*np.exp(-a[1]*x)*x**a[0]]
	
def f(x,y,a):
	return 0.5*np.sum((y-g(a,x))**2)	
	
def grad_f(x,y,a):
	return [-np.sum((y-g(a,x))*grad_g(x,a)[0]), -np.sum((y-g(a,x))*grad_g(x,a)[1])]
	
def hess_f(x,a):
	return np.matrix([[np.sum(grad_g(x,a)[0]**2), np.sum(grad_g(x,a)[0]*grad_g(x,a)[1])],[np.sum(grad_g(x,a)[0]*grad_g(x,a)[1]), np.sum(grad_g(x,a)[1]**2)]])

X = np.linspace(0.01,5,500)
print X[0:10]


b = 0.005
a = [2,3]
def jeu(a, b, g, X):
	res = []
	for x in X: 
		res.append(g(a,x)+b*np.random.randn())
	return res

donnes = jeu(a,b,g,X)
'''plt.plot(X,donnes, '+')
gs = [g(a,x) for x in X]
plt.plot(X, gs)
plt.show()
print len(grad_g(X, a)[0])
print grad_f(X, donnes, a)
print hess_f(X, donnes, a)'''


def leven(x, y, dmin):
	a = [1.5, 1.5]
	l = 0.001
	cout = f(x,y,a)
	k = 0
	d = dmin
	dlist = []
	llist = []
	llist.append(l)
	#alist = []
	#alist.append(a)
	flist = []
	flist.append(cout)
	while (k<400 and np.linalg.norm(d) >= dmin) :	
		coutav = cout
		grad = grad_f(x,y,a)
		hess = hess_f(x, a)
		hlm = hess*(1+l)
		d = - np.dot(grad, np.linalg.inv(hlm))
		d = d.tolist()[0]
		#print [a[0]-d[0], a[1]-d[1]]
		cout = f(x,y,[a[0]-d[0], a[1]-d[1]])
		k = k+1
		if(cout < coutav):
			a = [a[0]-d[0], a[1]-d[1]]
			l = l/10
			
		else:
			l *=10
			
		dlist.append(np.linalg.norm(d))
		flist.append(cout)
		#alist.append(a)
		llist.append(l)
	
	
	plt.plot(range(0,k), dlist)
	plt.show()
	#plt.plot(range(0,k+1), alist)
	#plt.show()
	plt.plot(range(0,k+1), llist)
	plt.show()
	plt.plot(range(0,k+1), flist)
	plt.show()
	print "convergence vers ", a, " au bout de ", k, " iterations."
	#return a

leven(X, donnes, 0.001)
