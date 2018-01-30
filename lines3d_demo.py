#Imports from the matplotlib library
import matplotlib as mpl
mpl.use('qt4agg')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


#--------------------------------------

#Definition of what to plot
fig = plt.figure() #opens a figure environment
ax = fig.gca(projection='3d') #to perform a 3D plot


theta = np.linspace(-4 * np.pi, 4 * np.pi, 100) #creates a linear interval
z = np.linspace(-2, 2, 100) #creates a linear interval
r = z**2 + 1 #defines the radial variable
x = r * np.sin(theta) #defines the x variable
y = r * np.cos(theta) #defines the y variable
ax.plot(x, y, z, label='parametric curve') #plot definition and options 
ax.legend() #adds a legend
mpl.rcParams['legend.fontsize'] = 10 #sets the legend font size

#Runs the plot command
plt.show()


