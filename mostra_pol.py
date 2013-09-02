
#-----------------------------
# Autor: Gustavo Frizzo
# Email: guto.smo@gmail.com
# 2011
#-----------------------------

from Tkinter import *
from numpy import *

#####################
#
# Funcao chamada por cria_pontos.py
#
#####################

def visualiza(vini, vfin, liga, img, img2):

	top = Toplevel()
	
	imgg = img
	imgg2 = img2
	
	w = imgg.width()*2+60
	h = imgg.height()+40
	# get screen width and height
	ws = top.winfo_screenwidth()
	hs = top.winfo_screenheight()
	# calculate position x, y
	x = (ws/2) - (w/2) 
	y = (hs/2) - (h/2)
	top.geometry('%dx%d+%d+%d' % (w, h, x, y))
	#top.geometry("1000x600")
	

	pp = Canvas(top, width = imgg.width()+15, height = imgg.height()+15)
	pp.place(x=20,y=20)
	pp.create_image(0, 0, anchor=NW, image=imgg)

	pp2 = Canvas(top, width = imgg.width()+15, height = imgg.height()+15)
	pp2.place(x=imgg.width()+20+20,y=20)
	pp2.create_image(0, 0, anchor=NW, image=imgg2)

	for i in range(len(vini)):
		vini[i] = [vini[i][1], vini[i][0]]
		vfin[i] = [vfin[i][1], vfin[i][0]]
		if vini[i][0] == 0:
			vini[i][0] = 1
		if vini[i][1] == 0:
			vini[i][1] = 1
		if vfin[i][0] == 0:
			vfin[i][0] = 1
		if vfin[i][1] == 0:
			vfin[i][1] = 1
		

	for i in liga:
		id = pp.create_polygon([vini[i[0]], vini[i[1]], vini[i[2]]], fill='', outline='red')
		id = pp2.create_polygon([vfin[i[0]], vfin[i[1]], vfin[i[2]]], fill='', outline='red')

	#Por algum motivo desconhecido, eh necessario voltar do jeito que estava....
	for i in range(len(vini)):
		vini[i] = [vini[i][1], vini[i][0]]
		vfin[i] = [vfin[i][1], vfin[i][0]]
		if vini[i][0] == 1:
			vini[i][0] = 0
		if vini[i][1] == 1:
			vini[i][1] = 0
		if vfin[i][0] == 1:
			vfin[i][0] = 0
		if vfin[i][1] == 1:
			vfin[i][1] = 0

	top.mainloop()
	
	
###########################
#
# Funcao para ser executada na forma dos comentarios abaixo 
#
###########################	
def visualiza2(vini, vfin, liga, img, img2):

	top = Toplevel()
	
	imgg = PhotoImage(file=img)
	imgg2 = PhotoImage(file=img2)
	
	w = imgg.width()*2+60
	h = imgg.height()+40
	# get screen width and height
	ws = top.winfo_screenwidth()
	hs = top.winfo_screenheight()
	# calculate position x, y
	x = (ws/2) - (w/2) 
	y = (hs/2) - (h/2)
	top.geometry('%dx%d+%d+%d' % (w, h, x, y))
	#top.geometry("1000x600")
	

	pp = Canvas(top, width = imgg.width()+15, height = imgg.height()+15)
	pp.place(x=20,y=20)
	pp.create_image(0, 0, anchor=NW, image=imgg)

	pp2 = Canvas(top, width = imgg.width()+15, height = imgg.height()+15)
	pp2.place(x=imgg.width()+20+20,y=20)
	pp2.create_image(0, 0, anchor=NW, image=imgg2)


	for i in range(len(vini)):
		vini[i] = [vini[i][1], vini[i][0]]
		vfin[i] = [vfin[i][1], vfin[i][0]]
		if vini[i][0] == 0:
			vini[i][0] = 1
		if vini[i][1] == 0:
			vini[i][1] = 1
		if vfin[i][0] == 0:
			vfin[i][0] = 1
		if vfin[i][1] == 0:
			vfin[i][1] = 1
		

	for i in liga:
		id = pp.create_polygon([vini[i[0]], vini[i[1]], vini[i[2]]], fill='', outline='red')
		id = pp2.create_polygon([vfin[i[0]], vfin[i[1]], vfin[i[2]]], fill='', outline='red')

	#Por algum motivo desconhecido, eh necessario voltar do jeito que estava....
	for i in range(len(vini)):
		vini[i] = [vini[i][1], vini[i][0]]
		vfin[i] = [vfin[i][1], vfin[i][0]]
		if vini[i][0] == 1:
			vini[i][0] = 0
		if vini[i][1] == 1:
			vini[i][1] = 0
		if vfin[i][0] == 1:
			vfin[i][0] = 0
		if vfin[i][1] == 1:
			vfin[i][1] = 0

	top.mainloop()

'''
vini = [[0, 0], [0, 295], [399, 295], [399, 0], [56, 79], [28, 122], [25, 159], [36, 197], [45, 231], [85, 240], [78, 165], [78, 204], [96, 57], [87, 102], [123, 124], [126, 195], [143, 108], [137, 124], [146, 141], [154, 124], [148, 173], [141, 192], [150, 203], [153, 190], [177, 135], [174, 158], [182, 179], [192, 157], [199, 120], [202, 154], [207, 180], [216, 151], [132, 158], [119, 245], [150, 239], [196, 219], [235, 193], [139, 60], [171, 65], [205, 91], [247, 132], [256, 170], [247, 77], [263, 39], [325, 55], [353, 135], [318, 240], [273, 243], [283, 281], [232, 178], [225, 58], [10, 142], [222, 26]]

vfin = [[0, 0], [0, 295], [399, 295], [399, 0], [70, 40], [99, 77], [120, 127], [120, 179], [80, 253], [119, 250], [161, 155], [155, 212], [126, 50], [151, 96], [180, 120], [181, 203], [205, 103], [191, 120], [204, 135], [219, 118], [210, 182], [193, 198], [203, 213], [220, 200], [238, 144], [234, 163], [242, 175], [250, 162], [273, 136], [261, 161], [273, 177], [272, 159], [189, 159], [170, 240], [209, 234], [237, 229], [275, 208], [183, 55], [225, 65], [261, 76], [298, 137], [300, 183], [279, 45], [317, 25], [334, 58], [347, 155], [308, 231], [267, 254], [285, 279], [281, 188], [245, 34], [52, 146], [246, 10]]

liga = [array([ 0,  3, 52,  0]), array([52,  3, 43, 52]), array([ 3, 44, 43,  3]), array([ 0, 52, 12,  0]), array([12, 52, 37, 12]), array([43, 50, 52, 43]), array([37, 52, 38, 37]), array([50, 38, 52, 50]), array([43, 42, 50, 43]), array([ 0, 12,  4,  0]), array([38, 50, 39, 38]), array([ 4, 12, 13,  4]), array([42, 39, 50, 42]), array([38, 16, 37, 38]), array([44, 42, 43, 44]), array([37, 13, 12, 37]), array([13, 37, 16, 13]), array([13, 16, 14, 13]), array([14, 16, 17, 14]), array([17, 16, 19, 17]), array([39, 16, 38, 39]), array([39, 19, 16, 39]), array([39, 28, 19, 39]), array([0, 4, 5, 0]), array([28, 24, 19, 28]), array([42, 40, 39, 42]), array([28, 39, 40, 28]), array([17, 19, 18, 17]), array([13,  5,  4, 13]), array([ 3, 45, 44,  3]), array([40, 31, 28, 40]), array([24, 18, 19, 24]), array([31, 29, 28, 31]), array([14, 17, 18, 14]), array([24, 28, 29, 24]), array([29, 27, 24, 29]), array([14, 18, 32, 14]), array([51,  5,  6, 51]), array([27, 25, 24, 27]), array([44, 40, 42, 44]), array([ 0,  5, 51,  0]), array([18, 24, 25, 18]), array([40, 44, 45, 40]), array([ 5, 13, 10,  5]), array([14, 10, 13, 14]), array([32, 18, 20, 32]), array([25, 20, 18, 25]), array([ 6,  5, 10,  6]), array([40, 41, 31, 40]), array([32, 10, 14, 32]), array([41, 49, 31, 41]), array([25, 27, 26, 25]), array([31, 30, 29, 31]), array([27, 29, 30, 27]), array([49, 30, 31, 49]), array([26, 27, 30, 26]), array([20, 25, 26, 20]), array([45, 41, 40, 45]), array([23, 21, 20, 23]), array([26, 23, 20, 26]), array([32, 20, 15, 32]), array([21, 15, 20, 21]), array([41, 36, 49, 41]), array([ 6, 10,  7,  6]), array([10, 32, 15, 10]), array([23, 22, 21, 23]), array([30, 49, 36, 30]), array([ 7, 10, 11,  7]), array([15, 11, 10, 15]), array([26, 30, 35, 26]), array([15, 21, 22, 15]), array([26, 22, 23, 26]), array([36, 35, 30, 36]), array([22, 26, 35, 22]), array([ 7, 11,  8,  7]), array([45, 46, 41, 45]), array([36, 41, 47, 36]), array([46, 47, 41, 46]), array([15, 22, 34, 15]), array([35, 34, 22, 35]), array([34, 33, 15, 34]), array([15,  9, 11, 15]), array([ 8, 11,  9,  8]), array([ 9, 15, 33,  9]), array([51,  6,  7, 51]), array([35, 36, 47, 35]), array([46, 48, 47, 46]), array([45,  2, 46, 45]), array([8, 1, 7, 8]), array([51,  7,  1, 51]), array([9, 1, 8, 9]), array([35, 47, 48, 35]), array([48, 46,  2, 48]), array([34, 35, 48, 34]), array([ 3,  2, 45,  3]), array([33,  1,  9, 33]), array([33, 34, 48, 33]), array([48,  1, 33, 48]), array([ 0, 51,  1,  0]), array([ 1, 48,  2,  1])]


print 'Pontos:',len(vini)
print 'Poligonos:',len(liga)

visualiza2(vini,vfin,liga, 'Imagens/ian.pgm', 'Imagens/cat2.pgm')
'''
