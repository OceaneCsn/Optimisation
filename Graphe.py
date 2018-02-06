#import pandas as pd

#data = pd.read_csv("droso_orthologies_filtered.txt", sep = " ", header = 0)
data  = open("droso_orthologies_filtered.txt", 'r')
lines = data.readlines()

ordre = []
sens = []
for l in lines:
	ordre.append(l.split(' ')[0])
	sens.append(l.split(' ')[5].split('\n')[0])

graph = dict()

vertices = range(1,804)
sommets = []
for v in vertices:
	sommets.append([str(v)+".d",str(v)+".f"])
	#sommets.append(str(v)+".f")

#print sommets

for v in sommets:
	graph[v[0]]=[]
	graph[v[1]]=[]

for i,v in enumerate(sommets):
	
	if(i<len(sommets)-1):
		graph[v[1]].append(sommets[i+1][0])
		graph[sommets[i+1][0]].append(v[1])

graph["Debut"] = []
graph["Fin"] = []
graph["Debut"].append(sommets[0][0])
graph["Fin"].append(sommets[len(sommets)-1][1])

print graph["Debut"]
print graph["Fin"]

###########################################################
vertices1 = ordre[1:]
sommets = []

for v in vertices1:
	sommets.append([str(v)+".d",str(v)+".f"])
	#sommets.append(str(v)+".f")

sens = sens[1:]
for i,v in enumerate(sommets):
	
	if(i<len(sommets)-1):
		
		if(int(sens[i])==1 and int(sens[i+1])==1):
			print "hey"
			graph[v[1]].append(sommets[i+1][0])
			graph[sommets[i+1][0]].append(v[1])
		
		if(int(sens[i])==-1 and int(sens[i+1])==-1):
			print "hey"
			graph[v[0]].append(sommets[i+1][1])
			graph[sommets[i+1][1]].append(v[0])
			
		if(int(sens[i])==1 and int(sens[i+1])==-1):
			print "hey"
			graph[v[1]].append(sommets[i+1][1])
			graph[sommets[i+1][1]].append(v[1])
			
		if(int(sens[i])==-1 and int(sens[i+1])==1):
			print "hey"
			graph[v[0]].append(sommets[i+1][0])
			graph[sommets[i+1][1]].append(v[1])


graph["Debut"].append(sommets[0][0])
graph["Fin"].append(sommets[len(sommets)-1][1])
print graph


def composantes(g):
	comp = 0
	marq = []
	while True:
		for s in g.keys():
			for v in s:
				if (v not in marq): 
