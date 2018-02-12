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



def methode_LM(a0, x, y, f, gradf, hessf, lam0=0.001, kmax=100, stopd=10**(-15), precision = 0.000001,  rall=False, print_step=True):
  lam = lam0
  k = 0

  a = a0
  last_a = np.add(a0,10) # to stop from trigering convergence at step 1
  d = 1
  as_ = [a]
  ds_ = []
  la_ = [lam]
  fs = []

  while k < kmax and np.linalg.norm(d)>stopd :#and abs(f(x, y, a)-f(x, y, last_a))>precision:
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
      print ("k : ",k)


    if f(x, y, at) < f(x, y, a):
      a = at
      lam *= 0.1
    else :
      lam *= 10
    as_.append(a)
    ds_.append(d)
    la_.append(lam)
    fs.append(f(x, y, at))

    k+=1
  if rall :
    return as_, ds_, la_, fs
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







#ax.plot(x, y, '+', color = "blue")
#ax.plot(x, g(x, a), color='red', linewidth=2)



# initial conditions

a0 = (1.5,1.5)

a = (2.0,3.0)
b = 0.01
x, y1 = data(a, b)
x, y2 = data(a, 0.05)
x, y3 = data(a, 0.1)
x, y4 = data(a, 0.5)
x, y5 = data(a, 0.7)

af1, k1 = methode_LM(a0, x, y1, f, gradf, hessf, kmax = 100, precision = 10**(-10),  rall = False, print_step=True)
af2, k2 = methode_LM(a0, x, y2, f, gradf, hessf, kmax = 100, precision = 10**(-10),  rall = False, print_step=True)
af3, k3 = methode_LM(a0, x, y3, f, gradf, hessf, kmax = 100, precision = 10**(-10),  rall = False, print_step=True)
af4, k4 = methode_LM(a0, x, y4, f, gradf, hessf, kmax = 100, precision = 10**(-10),  rall = False, print_step=True)
af5, k5 = methode_LM(a0, x, y5, f, gradf, hessf, kmax = 100, precision = 10**(-10),  rall = False, print_step=True)

az, dz1, lz1, fz1 = methode_LM(a0, x, y1, f, gradf, hessf,  precision = 10**(-10), kmax = 50,rall = True)
az, dz2, lz2, fz2 = methode_LM(a0, x, y2, f, gradf, hessf,  precision = 10**(-10), kmax = 50,rall = True)
az, dz3, lz3, fz3 = methode_LM(a0, x, y3, f, gradf, hessf,  precision = 10**(-10), kmax = 50,rall = True)
az, dz4, lz4, fz4 = methode_LM(a0, x, y4, f, gradf, hessf,  precision = 10**(-10), kmax = 50,rall = True)
az, dz5, lz5, fz5 = methode_LM(a0, x, y5, f, gradf, hessf,  precision = 10**(-10), kmax = 50,rall = True)
# EVERYTHING ELSE

'''#Definition of what to plot
fig = plt.figure() #opens a figure environment
#ax = fig.gca(projection='3d') #to perform a 3D plot
ax = fig.gca() #to perform a 3D plot
X = np.arange(-5, 5.0, 0.25) #x range
Y = np.arange(-5, 5.0, 0.25) #y range
X, Y = np.meshgrid(X, Y) #creates a rectangular grid on which to plot the function values (Z)

dz1 = data_manip(dz1)
dz2 = data_manip(dz2)
dz3 = data_manip(dz3)
dz4 = data_manip(dz4)
dz5 = data_manip(dz5)'''

'''ax.plot(range(len(lz1)), lz1, color = "purple", label = 'b = 0.001')
ax.plot(range(len(lz2)), lz2, color = "blue", label = 'b = 0.005')
ax.plot(range(len(lz1)),lz3, color = "green", label = 'b = 0.01')
#ax.plot(range(len(lz1)), lz4, color = "orange", label = 'b = 0.05')
#ax.plot(range(len(lz1)), lz5, color = "red", label = 'b = 0.1')
plt.legend()

'''
az1, az2 = data_manip2(az)


'''fig2 = plt.figure()
plt.title("Fonction f de cout au cours du temps")
ax2 = fig2.gca()
ax2.plot(range(len(fz)), fz)
#ax2.plot(range(len(az2)), az2, color='green')

#dz1 = data_manip(dz)'''

fig3 = plt.figure()
plt.title("Fonction g pour b = 0.01")
ax3 = fig3.gca()
ax3.plot(x, g(x,af1), color = "purple", label = 'b = 0.01')
ax3.plot(x, y1, '+', color = "purple")

fig5 = plt.figure()
plt.title("Fonction g pour b = 0.05")
ax5 = fig5.gca()
ax5.plot(x, g(x,af2),color = "blue")
ax5.plot(x, y2, '+', color = "blue")
fig4 = plt.figure()
plt.title("Fonction g pour b = 0.1")
ax4 = fig4.gca()
ax4.plot(x, g(x,af3), color = "green", label = 'b = 0.1')
ax4.plot(x, y3, '+', color = "green")
fig2 = plt.figure()
plt.title("Fonction g pour b = 0.5")
ax2 = fig2.gca()
ax2.plot(x, g(x,af4), color = "orange", label = 'b = 0.5')
ax2.plot(x, y4, '+', color = "orange")
fig = plt.figure()
plt.title("Fonction g pour b = 0.7")
ax = fig.gca()
ax.plot(x, g(x,af5), color = "red", label = 'b = 0.7')
ax.plot(x, y5, '+', color = "red")
#plt.legend()
plt.show()
#ax3.plot(range(len(dz1)), dz1)
'''
fig4 = plt.figure()
plt.title("Valeur de lambda")
ax4 = fig4.gca()
ax4.plot(range(len(lz)), lz)


#plot des courbes de tous les a
for i,ai in enumerate(az[1:]):
  ax.plot(x, g(x,ai), color=( 0.7-0.01*i, 0.7-0.01*i, 0.7-0.01*i), linewidth=1)


# plot du a final
#ax.plot(x, g(x, a), color='red', linewidth=2)
ax.plot(x, g(x, af), color='darkgreen', linewidth=2)

#plt.title("Fonction g au fil des itérations")
'''


plt.show()
