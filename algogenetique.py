import matplotlib as mpl
mpl.use('qt4agg')
import random as rd
import matplotlib.pyplot as plt
import numpy as np


def genome(T):
    res = []
    for t in range(T):
        res.append(rd.sample([-1,1],1)[0])
    return res
    

def cout(genome, r0):
    pos = 0
    poslist = [0]
    cout = 0
    for g in genome:
        pos += g
        poslist.append(pos)
        if(abs(pos) > r0):
            cout+=1
    #print("cout : ",cout)
    '''plt.plot(range(len(poslist)),poslist)
    plt.axhline(y = r0, color = "red")
    plt.axhline(y = -r0, color = "red")
    plt.show()'''
    return cout

g = genome(100)
#cout(g, 4)

def population(N, T):
    return [genome(T) for n in range(N)]

def tripop(pop, r0):
    return(np.argsort([cout(genome, r0) for genome in pop]) )
    
def selection(pop, r0, prop):
    pop = pop
    ordre = tripop(pop, r0)
    ordre = ordre.tolist()
    sel = []
    for i in range(int(len(pop)*prop)):
        a = max(ordre)
        b = ordre.index(a)
        suppr = ordre.pop(b)
        
        sel.append(pop[suppr])
        pop.pop(b)
    return [sel, pop]

def mutation(genome, Tm):
    res = genome
    for i,g in enumerate(res):
        r = rd.random()
        if(r <= Tm):
            res[i] = -res[i]
    return res
    
def mutationPop(pop, Tm):
    print(pop)
    return [mutation(genome, Tm) for genome in pop]
    
def Croisement(pop, Tc):    
    res = pop
    for i,ind in enumerate(res):
        r = rd.random()
        if(r <= Tc):
            pos = rd.randint(0,len(ind)-1)
            posicrois = rd.randint(0, len(res)-1)
            print("avant : ",res[i], res[posicrois])
            print("pos : ", pos)
            tmp = res[i][pos:]
            res[i][pos:] = res[posicrois][pos:]            
            res[posicrois][pos:] = tmp
            print("apres : ",res[i], res[posicrois])
    return res
            
            
def iteration(pop, Tm, Tc, r0, prop):
    poptri = selection(pop, r0, prop)[0]
    sel = poptri[0]
    lereste = poptri[1]
    sel = mutationPop(sel, Tm)
    sel = Croisement(sel, Tc)
    return sel+lereste
    
#print(len(selection(population(100,10),3, 0.5)))
#print(mutationPop(population(10, 5), 0.2))
print(iteration(population(10, 10), 0.2, 0.2, 2, 0.5))
