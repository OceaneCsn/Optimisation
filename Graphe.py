import pandas as pd

data = pd.read_csv("droso_orthologies_filtered.txt", sep = " ", header = 0)

graph = dict()

vertices = range(1,803)
vin = []
vout = []
for v in vertices:
	vin.append(v+0.1)
	vout.append(v+0.2)

v = vin+vout
v.sort()

for vertex in v:
	graph[vertex]=[]

for k in graph.keys():
	#print k, ((k-0.1)*10)%10
	if(((k-0.1)*10)%10==0):
		graph[k].append(k+0.1)
		print k+0.1
	else:
		graph[k]=
		
print graph
#on relie les vertex par ordre croissant de out vers in pour melanogaster

