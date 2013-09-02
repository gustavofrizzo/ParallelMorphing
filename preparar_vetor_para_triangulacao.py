
#-----------------------------
# Autor: Gustavo Frizzo
# Email: guto.smo@gmail.com
# 2011
#-----------------------------

from numpy import *
from matplotlib.tri import *

def Triangulacao(a):
	x = []
	y = []

	for i in a:
		x.append(i[0])
		y.append(i[1])
	
	t = Triangulation(x, y)
	res = t.triangles

	res2 = []

	for i in range(len(res)):
		res2.append(append(res[i],res[i][0]))

	return res2

