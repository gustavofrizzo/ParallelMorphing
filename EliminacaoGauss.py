# -*- coding: latin1 -*-


# Autor: Jonnathan Weber
# Email: jonny172@Msn.com
# Data: 28/05/2010
# Objetivo: Calcular sistemas matriciais a partir do metodo de gauss-compacto

from fractions import Fraction as f

'''
Incognitas = 1
matrizInicial = [['2', '-4'], ['2', '3']]
matrizIndependentesInicial = [['2', '21']]
ordem = 2
resultado = EliminacaoGauss(ordem, Incognitas, matrizInicial, matrizIndependentesInicial)
resultado.getMatrizFinal()
'''
class EliminacaoGauss():
	erro = 0 
	def __init__(self, ordem, incognitas, matriz1, matriz2):
		self.ordem = ordem
		self.qIncognitas = incognitas
		self.matrizInicial = matriz1
		self.matrizIndependentesInicial = matriz2
		self.calculaMatrizMetodoEliminacao()
		
	def getErro(self):
		return self.erro
		
	def setErro(self,_nEr):
		self.erro = _nEr
		
	def defineMatrizAuxiliar(self):
		matriz = []
		for i in range(self.qIncognitas):
			linha = []
			for j in range(self.ordem):
				linha.append(self.matrizIndependentesInicial[i][j])
			matriz.append(linha)
		return matriz
	
	def defineMatrizIcognitas(self):
		matriz = []
		for i in range(self.qIncognitas):
			linha = []
			for j in range(self.ordem):
				linha.append('0')
			matriz.append(linha)
		return matriz
		
	def defineMatrizFinal(self):
		matriz = []
		for i in range(self.ordem):
			linha = []
			for j in range(self.ordem):
				linha.append(self.matrizInicial[i][j])
			matriz.append(linha)
		return matriz
		
	def calculaMatrizMetodoEliminacao(self):
		self.matrizFinal = self.defineMatrizFinal()
		self.matrizIndependentesFinal = self.defineMatrizAuxiliar()
		self.matrizIndependentesFinal[0][0] = self.matrizIndependentesFinal[0][0]
		k = self.ordem
		lista = []
		nI = 0
		while k > 0:
			i = self.ordem - k
			den = f(self.matrizFinal[self.ordem-k][self.ordem-k])
			#Verifica elemento e permuta a linha
			if (den == 0):
				if (i == 0):
					self.setErro(1)
					break
				else:
					lista = self.matrizFinal[i+1]
					self.matrizFinal[i+1] = self.matrizFinal[i]
					self.matrizFinal[i] = lista
					nI = self.matrizIndependentesFinal[0][i+1]
					self.matrizIndependentesFinal[0][i+1] = self.matrizIndependentesFinal[0][i]
					self.matrizIndependentesFinal[0][i] = nI
			den = f(self.matrizFinal[self.ordem-k][self.ordem-k])
			if (den == 0):
				self.setErro(1)
				break
			while i+1 < self.ordem:
				contador = 0
				j = self.ordem - k + 1
				num = f(self.matrizFinal[i+1][self.ordem-k])
				divisor = num/den
				while contador < self.ordem:
					if (j > self.ordem):
						break
					self.matrizFinal[i+1][j-1] = f(self.matrizFinal[i+1][j-1]) - f(self.matrizFinal[self.ordem-k][j-1]) * divisor
					j += 1
					contador += 1
				self.matrizIndependentesFinal[0][i+1] = f(self.matrizIndependentesFinal[0][i+1]) - f(self.matrizIndependentesFinal[0][self.ordem-k]) * divisor
				i += 1
			k -= 1
		else:
			self.calculaXMetodoEliminacao()
		
	def calculaXMetodoEliminacao(self):
		self.matrizIncognitas = self.defineMatrizIcognitas()
		c = 0
		k = self.ordem-1
		while c < self.ordem:
			termo = self.calculaX(self.matrizFinal,self.matrizIncognitas[0],k-c,k)
			if (f(self.matrizFinal[k-c][k-c]) == 0):
				self.setErro(1)
				break
			else:
				self.matrizIncognitas[0][k-c] = (f(self.matrizIndependentesFinal[0][k-c]) - f(termo))/f(self.matrizFinal[k-c][k-c])
				c += 1
		
			
	def calculaX(self, mA,x,j,k):
		if (j == k):
			k -= 1
		if (k < 0):
			return 0	
		if (k < j):
			return 0
		return f(mA[j][k])*f(x[k]) + self.calculaX(mA,x,j,k-1)
		
	def getMatrizFinal(self):
		return self.matrizFinal
	
	def getMatrizIndependentesFinal(self):
		return self.matrizIndependentesFinal
	
	def getMatrizIncognitas(self):
		return self.matrizIncognitas
