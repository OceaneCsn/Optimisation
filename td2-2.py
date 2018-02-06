#coding:utf-8

#Imports from the matplotlib library

import matplotlib as mpl
mpl.use('qt4agg')
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
#--------------------------------------



def g(x,a):
  return (x**a[0])*np.exp(-x*a[1])



# For data generation with gaussian noise
def data(a,b):
  x = pl.frange(0,3,0.01)[1:]
  y = g(x, a)+b*np.random.randn(len(x))
  return x, y


def gp(x, a):
  return (np.log(x)*(x**a[0])*np.exp(-x*a[1]),
          (-x**(a[0]+1))*np.exp(-x*a[1]-1))

# Cost function to minimize (nonlinear regression)
def f(x, y, a):
  return 0.5*sum((y-g(x, a))**2)


# Gradient
def gradf(x, y, a):
  return [-sum((y-g(x, a))*gp(x, a)[i]) for i in range(len(a))]

# 2nd derivative
def d2f(x, y, a):
  return (sum(gp(x, a)[0]*gp(x, a)[0]), sum(gp(x, a)[0]*gp(x, a)[1]), sum(gp(x, a)[1]*gp(x, a)[0]), sum(gp(x, a)[1]*gp(x, a)[1]))

# Hessian
def hessf(x, y, a):
  d = d2f(x, y, a)
  return np.matrix([[d[0], d[1]],
                    [d[2], d[3]]])


# direction
def direc(grad, hlm):
  return np.dot(np.linalg.inv(hlm), grad).dot(-1)



def HLM(H, lamb):
  l = len(H)
  T = np.dot(np.eye(2), lamb)
  for i in range(l):
    for j in range(l):
      H[i,j] = H[i,j]*(1+T[i,j])
  return H



def methode_LM(a0, x, y, f, gradf, hessf, lam0=0.001, kmax=100, stopd=10**(-15), precision = 0.0001,  rall=False, print_step=False):
  lam = lam0
  k = 0

  a = a0
  last_a = np.add(a0,10) # to stop from trigering convergence at step 1
  d = 1
  as_ = [a]
  ds_ = []
  la_ = [lam]

  while k < kmax :#and np.linalg.norm(d)>stopd and abs(f(x, y, a)-f(x, y, last_a))>precision:
    last_a = a
    G = gradf(x, y, a)
    H = HLM(hessf(x, y, a), lam)
    
    d = direc(G, H)
    at = np.add(a,d)
    at = (at[0,0], at[0,1]) #data formating...
    
    if print_step:
      print ("\na : ", at)
      print ("d : ",d)
      print ("G : ",G)
      print ("H : ",H)
      print ("lam : ",lam)


    if f(x, y, at) < f(x, y, a):
      a = at
      lam *= 0.1
    else :
      lam *= 10
    as_.append(a)
    ds_.append(d)
    la_.append(lam)

    k+=1
  if rall :
    return as_, ds_, la_
  else :
    return a, k


# from list of matrix to 2 lists of values
def data_manip(d):
  r = []
  for o in d:
    
    r.append(np.linalg.norm(o[0]))
    
  return r


# from list of tuples to 2 lists of values
def data_manip2(d):
  r = []
  s = []
  for o in d:
    r.append(o[0])
    s.append(o[1])
  return r,s



#initial data

a = (2.0,3.0)
b = 0.01

x, y = data(a, b)


#Definition of what to plot
fig = plt.figure() #opens a figure environment
#ax = fig.gca(projection='3d') #to perform a 3D plot
ax = fig.gca() #to perform a 3D plot
X = np.arange(-5, 5.0, 0.25) #x range
Y = np.arange(-5, 5.0, 0.25) #y range
X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)



ax.plot(x, y, '+')
#ax.plot(x, g(x, a), color='red', linewidth=2)



# initial conditions

a0 = (1.5,1.5)

af, k = methode_LM(a0, x, y, f, gradf, hessf, kmax = 100, precision = 10**(-10),  rall = False, print_step=True)

az, dz, lz = methode_LM(a0, x, y, f, gradf, hessf,  precision = 10**(-10), kmax = 50,rall = True)
# EVERYTHING ELSE


az1, az2 = data_manip2(az)

fig2 = plt.figure()
plt.title("convergence de a")
ax2 = fig2.gca()
ax2.plot(range(len(az1)), az1)
ax2.plot(range(len(az2)), az2, color='red')

dz1 = data_manip(dz)

fig3 = plt.figure()
plt.title("Norme de la direction de descente")
ax3 = fig3.gca()
ax3.plot(range(len(dz1)), dz1)

fig4 = plt.figure()
plt.title("Valeur de lambda")
ax4 = fig4.gca()
ax4.plot(range(len(lz)), lz)


#plot des courbes de tous les a
for i,ai in enumerate(az[1:]):
  ax.plot(x, g(x,ai), color=(0,0, 1-0.08*i), linewidth=1)


# plot du a final
#ax.plot(x, g(x, a), color='red', linewidth=2)
ax.plot(x, g(x, af), 'k--', color='deepblue', linewidth=4)





plt.show()
