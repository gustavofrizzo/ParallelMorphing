
#-----------------------------
# Autor: Gustavo Frizzo
# Email: guto.smo@gmail.com
# 2011
#-----------------------------

from adpil import *
from parallel_morph import *

#Execucao de exemplo

ini = adread('Imagens/x.pgm')
fin = adread('Imagens/y.pgm')


periodo = 10

processos = 10

#Pontos da Imagem Inicial
vini = [[0, 0], [0, 295], [399, 295], [399, 0], [173, 54], [48, 147], [168, 223], [285, 84], [281, 211]]

#Pontos da Imagem Final
vfin = [[0, 0], [0, 295], [399, 295], [399, 0], [154, 55], [57, 133], [161, 198], [277, 84], [293, 240]]

#Ligacoes que representam os poligonos criados pelos pontos
liga = [array([0, 3, 4, 0]), array([3, 7, 4, 3]), array([0, 4, 5, 0]), array([5, 4, 6, 5]), array([7, 6, 4, 7]), array([8, 6, 7, 8]), array([8, 7, 2, 8]), array([3, 2, 7, 3]), array([6, 1, 5, 6]), array([0, 5, 1, 0]), array([6, 8, 2, 6]), array([1, 6, 2, 1])]

#Executa a funcao com os parametros definidos.
morphing(ini, fin, vfin, vini, liga, periodo, processos)
