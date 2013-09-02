
#-----------------------------
# Autor: Gustavo Frizzo
# Email: guto.smo@gmail.com
# 2011
# Parallel Image Morphing
#-----------------------------

from numpy import *
from math import *

'''
Morfismo
'''
def inter_linear(a, b, t, periodo):  #t eh um tempo qlqr no periodo, a e b sao apenas numeros
    ###############
    # Funcao usada inter-imagens
    ###############
    if t == 0: return a
    if t == periodo-1: return b
    c = b - a
    c = float(c) / float((periodo-1))#Float para nao perder a precisao
    res = float(a)
    for i in range(0, t):
        res = res + c      
    return int(res)

#for i in range(10):
#    print inter_linear(0,500, i, 10)

def inter_linear_new(a, t, periodo):  #t eh um tempo qlqr no periodo, a eh uma tupla
    ###############
    # Funcao usada inter-imagens
    ###############
    if t == 0: return a[0]
    if t == periodo-1: return a[1]
    c = a[1] - a[0]
    c = c / (periodo-1)
    res = a[0]
    for i in range(0, t):
        res = res + c      
    return res

def cria_linha(pi,pf): #retorna um vetor de pontos que forma uma linha ligando os pontos --> cria_linha([0,0], [10,12])
    res = []
    distancia = ((pi[0]-pf[0])**2 + (pi[1]-pf[1])**2)**0.5 #pitagoras
    distancia = distancia +1
    for i in range(distancia):
        res.append ((ceil(inter_linear(pi[0],pf[0], i, distancia)),ceil(inter_linear(pi[1],pf[1], i, distancia))))#ceil arredonda pra cima
    return res
