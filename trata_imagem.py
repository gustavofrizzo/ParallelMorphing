
#-----------------------------
# Autor: Gustavo Frizzo
# Email: guto.smo@gmail.com
# 2011
# Parallel Image Morphing
#-----------------------------

from numpy import *
from ia636 import *
import Image
import ImageOps
import ImageFilter


def minRet(vet):
    #Movido de TCC2
    #Retorna valores para criar um retangulo minimo de um triangulo qualquer
    #e a posicao que ele deve iniciar o mapeamento
    #Cria um retangulo minimo para impedir que pontos desnecessarios da imagem sejam visitados
    
    #vet = [[0,0],[10,15],[20,50]]
    aux1 = []
    aux2 = []
    a = []
    b = []
    for i in vet:
        aux1.append(i[0])
        aux2.append(i[1])
    a = [abs(aux1[0] - aux1[1]),abs(aux1[1] - aux1[2]),abs(aux1[2] - aux1[0])]
    b = [abs(aux2[0] - aux2[1]),abs(aux2[1] - aux2[2]),abs(aux2[2] - aux2[0])]
    return zeros([max(a),max(b)]), [min(aux1),min(aux2)]


def tratarImagem(img):
	img = naolinear(img, (3,3), 'maximo') #Ainda eh meio lento =/
	#img = naolinearNormal(img, (3,3), 'mediana')
	a = Image.fromarray(img)
	#a = a.filter(ImageFilter.MaxFilter)
	a = a.filter(ImageFilter.MedianFilter)
	a = ImageOps.autocontrast(a)
	img = array(a)
	return img


def naolinear(f, dim_janela, op='minimo'):
	A, B = dim_janela
	g = zeros(f.shape, 'uint8')
	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			if f[i, j] == 0:
				lin_ini = i - A/2
				lin_fim = lin_ini + A
				col_ini = j - B/2
				col_fim = col_ini + B
				if (lin_ini < 0): lin_ini = 0
				if (lin_fim >= f.shape[0]): lin_fim = f.shape[0]
				if (col_ini < 0): col_ini = 0
				if (col_fim >= f.shape[1]): col_fim = f.shape[1]
				m = f[lin_ini:lin_fim, col_ini:col_fim]
				if op=='maximo':
					g[i,j] = max(ravel(m))
				elif op=='minimo':
					g[i,j] = min(ravel(m))
				elif op=='mediana':
					lista = sort(ravel(m))
					g[i,j] = lista[len(lista)/2]
			else:
				g[i,j] = f[i,j]
	return g
	
def naolinearNormal(f, dim_janela, op='minimo'):
	A, B = dim_janela
	g = zeros(f.shape, 'uint8')
	for i in range(f.shape[0]):
		for j in range(f.shape[1]):
			lin_ini = i - A/2
			lin_fim = lin_ini + A
			col_ini = j - B/2
			col_fim = col_ini + B
			if (lin_ini < 0): lin_ini = 0
			if (lin_fim >= f.shape[0]): lin_fim = f.shape[0]
			if (col_ini < 0): col_ini = 0
			if (col_fim >= f.shape[1]): col_fim = f.shape[1]
			m = f[lin_ini:lin_fim, col_ini:col_fim]
			if op=='maximo':
				g[i,j] = max(ravel(m))
			elif op=='minimo':
				g[i,j] = min(ravel(m))
			elif op=='mediana':
				lista = sort(ravel(m))
				g[i,j] = lista[len(lista)/2]
	return g
	
