import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#mpl.use('qt4agg')

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np


def g(a,x):
	return x**a[0]*np.exp(-a[1]*x)

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
plt.plot(X,donnes, '+')
gs = [g(a,x) for x in X]
plt.plot(X, gs)
plt.show()
