
#-----------------------------
# Autor: Gustavo Frizzo
# Email: guto.smo@gmail.com
# 2011
# Parallel Image Morphing
#-----------------------------

from adpil import *
from numpy import *
from morfismo import *
from mape import *
from trata_imagem import *
from multiprocessing import Process

'''
Funcao que retorna a posicao dos vertices em um dado instante do tempo
'''
def  morfa(vi, vf, t, periodo):
    ret = []
    for i in range(len(vi)):
        ret.append([inter_linear(vi[i][0],vf[i][0],t,periodo), inter_linear(vi[i][1],vf[i][1],t,periodo)])
    return ret


'''
Funcao responsavel por gerar o mapeando de cada imagem individualmente no tempo
'''
def morfa3(ini, vet, vet2, liga, q, periodo):
    ret = morfa(vet,vet2, q, periodo)
    mp = mapeia(ini, vet, ret, liga)
    return mp

'''
Funcao responsavel por salvar as imagens geradas 
'''
def morfa4(ini, fin, vfin, vini, liga, qtd, li, lf): #qtd = quantidade de periodos / ll = periodo que inicia o processo / lf = periodo que termina o processo
	for i in range(li,lf+1):
		if i == 0:
			mp = tratarImagem(ini)
			mf = morfa3(fin, vfin, vini, liga, (qtd-1)-i, qtd)
		elif i == (qtd-1):
			mp = morfa3(ini, vini, vfin, liga, i, qtd)
			mf = tratarImagem(fin)
		else:
			mp = morfa3(ini, vini, vfin, liga, i, qtd)
			mf = morfa3(fin, vfin, vini, liga, (qtd-1)-i, qtd)
		alpha = ((100 * i) / (qtd-1))/100.0
		out = (mp * (1.0 - alpha)) + (mf * alpha) #Pinta os pixels conforme posicao da imagem no tempo
		print 'Imagem', i+1,'Processada!'
		adwrite('results/r'+str(i)+'.pgm', out) #Resultado final
		adwrite('results/rrr'+str(i)+'.pgm', mp) #Deformacao da imagem inicial
		adwrite('results/rr'+str(i)+'.pgm', mf) #Deformacao da imagem final

		
def morphing(ini, fin, vfin, vini, liga, qtdPer, qtdProc): #qtd de processos deve ser maior que qtd de periodos
	print '************************************************'
	print '* Quantidade de Processos em Paralelo:', qtdProc
	print '* Quantidade de Imagens que Serao Geradas:', qtdPer
	print '* Resolucao da Imagem:', str(ini.shape), '(Altura X Largura) -',str(ini.shape[0]*ini.shape[1]),'pixels' 
	print '* Numero de Pontos:', str(len(vini))
	print '* Numero de Poligonos:', str(len(liga))
	print '************************************************'
	print 'Aguarde o termino do processamento...'
	print ''
	processos = []
	passo = 0
	esc = zeros(qtdProc, int)
	j = 0 
	for i in range(qtdPer):
		esc[j] = esc[j] + 1
		j = j + 1
		if j >= qtdProc:
			j = 0
	if qtdProc > qtdPer:
		print 'Quantidade de processos deve ser maior que quantidade de periodos!'
		return
		
	for i in range(qtdProc):
		if i == qtdProc-1 and (passo + qtdPer/qtdProc)-1 < qtdPer-1:
			processos.append(Process(target = morfa4, args = (ini, fin, vfin, vini, liga, qtdPer, passo, passo + esc[i] -1,)))
		else:
			processos.append(Process(target = morfa4, args = (ini, fin, vfin, vini, liga, qtdPer, passo, passo + esc[i] - 1,)))
		passo = passo + esc[i]
	
	for i in processos: #Inicia todos os processos
		i.start()
		
	for i in processos: #Aguarda o termino de todos os processos
		i.join()
		
	print '\nAs imagens foram salvas na pasta "results".'		
		
