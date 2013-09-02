#-----------------------------
# Autor: Gustavo Frizzo
# Email: guto.smo@gmail.com
# 2011
# Parallel Image Morphing
#-----------------------------

from numpy import *
from adpil import *
from EliminacaoGauss import *
from matplotlib.nxutils import *
from trata_imagem import *
import warnings

warnings.simplefilter('ignore', DeprecationWarning)

'''
Cria matriz para a funcao de Eliminacao de Gauss
'''
def cria_matriz(v1, v2):
	'''m1 = [[v1[0][0], v1[0][1], 1],  
		     [v1[1][0], v1[1][1], 1],
		     [v1[2][0], v1[2][1], 1]]'''
	
	m1 =[[v1[0][0], v1[0][1], 1],  
		 [v1[1][0], v1[1][1], 1],
		 [v1[2][0], v1[2][1], 1]]
	
	for i in range(3):
		for j in range(3):
			m1[i][j] = str(m1[i][j])
			
	m2 = [[v2[0][0], v2[1][0], v2[2][0]]]#m2 possui as coordenadas das linhas de cada vertice
	m3 = [[v2[0][1], v2[1][1], v2[2][1]]]#m3 possui as coordenadas das colunas de cada vertice
	
	for i in range(3):
		m2[0][i] = str(m2[0][i])
		m3[0][i] = str(m3[0][i])
	
	ret = [m1,m2,m3]
	return ret

'''
Calcula o Metodo de Eliminacao de Gauss
'''
def ElimGauss(v1, v2):
	matriz = cria_matriz(v1, v2)	
	incog = EliminacaoGauss(3, 1, matriz[0], matriz[1])
	incog = incog.getMatrizIncognitas()
	
	res = []
	res.append([float(incog[0][0]),float(incog[0][1]),float(incog[0][2])])
	
	incog = EliminacaoGauss(3, 1, matriz[0], matriz[2])
	incog = incog.getMatrizIncognitas()
	
	res.append([float(incog[0][0]),float(incog[0][1]),float(incog[0][2])])
	
	return res
	
'''
Funcao de mapeamento
'''
def mapeia(le, vet, vet2, liga):
	mat = zeros(le.shape) #imagem de retorno
	#varre todos os poligonos da imagem
	for l in liga:
		vert = [vet[l[0]], vet[l[1]] , vet[l[2]]]
		vert2 = [vet2[l[0]], vet2[l[1]] , vet2[l[2]]]
		
		for x in range(3): #Tratamento para nao ocorrer divisao por zero
			if vet[x][0] == 0: vet[x][0] = 0.001
			if vet[x][1] == 0: vet[x][1] = 0.001
		
		box = minRet(vert) #Cria um retangulo minimo para impedir que pontos desnecessarios da imagem sejam visitados
		calc = ElimGauss(vert,vert2)

		for i in range(box[1][0], box[1][0] + box[0].shape[0]+1):
			for j in range(box[1][1], box[1][1] + box[0].shape[1]+1):
				#points = array([[i,j]])
				if pnpoly(i, j, vert):
					xi = calc[0][0]*(i) + calc[0][1]*(j) + calc[0][2]#*i*j
					yi = calc[1][0]*(i) + calc[1][1]*(j) + calc[1][2]#*i*j
					mat[xi,yi] = le[i,j]
					#mat[i,j] = le[i,j]
	mat = tratarImagem(mat)
	return mat
