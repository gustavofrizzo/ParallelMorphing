
#-----------------------------
# Autor: Gustavo Frizzo
# Email: guto.smo@gmail.com
# 2011
#-----------------------------

from Tkinter import *
from preparar_vetor_para_triangulacao import *
from mostra_pol import *
import tkMessageBox

def center_window(w=300, h=200):
	ws = root.winfo_screenwidth()
	hs = root.winfo_screenheight()
	x = (ws/2) - (w/2) 
	y = (hs/2) - (h/2)
	root.geometry('%dx%d+%d+%d' % (w, h, x, y))

class MyButton(Button):

    def __init__(self, parent, xx, yy, tipo): #tipo referente a imagem 1 ou 2
        self.xx = xx
        self.yy = yy
        if tipo == 1:
            self.b = Button(parent, background = 'red', command = self.pos )
        else:
            self.b = Button(parent, background = 'red')            
        self.b.place_configure(x = xx, y = yy, width = 5, height = 5)

    def pos(self):
        global pol
        pol = pol + (self.xx+2, self.yy+2)
        if len(pol) >= 6:
            global p, p2, nmr_pol, vet_poligonos, vet
            vet_poligonos.append([vet.index([pol[1],pol[0]]) ,vet.index([pol[3],pol[2]]) , vet.index([pol[5],pol[4]]), vet.index([pol[1],pol[0]])])
            p.create_polygon(pol, fill='', outline='red')
            aux = ()
            for i in vet_poligonos[len(vet_poligonos)-1]:
                aux = aux + (vet2[i][1], vet2[i][0])
            p2.create_polygon(aux, fill='', outline='red')
            pol = ()
            nmr_pol = nmr_pol + 1
        print self.yy+2, self.xx+2
        return [self.yy+2, self.xx+2]
        

vet = []
vet2 = []
pol = ()
nmr_pol = 0
vet_poligonos = []
flag = 1
flag2 = 0 #flag que obriga a criacao de um ponto em cada imagem


def morfJanela():
    pass

def ImprimeIni():
    print vet
    
def ImprimeFin():
    print vet2
    
def ImprimeTudo():
    print ''
    print 'vini = ' + str(vet)
    print ''
    print 'vfin = ' + str(vet2)
    print ''
    print 'liga = ' + str(Triangulacao(vet))
    print ''

def Visu():
	if len(vet) == len(vet2):
		visualiza(vet, vet2, Triangulacao(vet), img, img2)
	else:
		tkMessageBox.showinfo("Atencao!", "Voce esqueceu de marcar \num ponto na segunda imagem!")
 
def ImprimeLiga():
    print vet_poligonos

def mudaFlag():
    global flag, blq
    if flag == 1:
        flag = 0
        blq.configure(text = "Desbloquear Criacao de Pontos")
    else:
        flag = 1
        blq.configure(text = "Bloquear Criacao de Pontos")

def Posicao(event):
    global flag2
    if flag == 1 and flag2 == 0:
        posicao.set('('+str(event.y)+', '+str(event.x)+')')
        but = MyButton(p, event.x-2, event.y-2, 1)
        vet.append([event.y,event.x])
        flag2 = 1



def Posicao2(event):
    global flag2
    if flag == 1 and flag2 == 1:
        posicao2.set('('+str(event.y)+', '+str(event.x)+')')
        id = p2.create_rectangle((event.x-2, event.y-2,event.x+2, event.y+2), fill="red", outline="#fb0", tags = 2)
        but2 = MyButton(p2, event.x-2, event.y-2, 2)
        vet2.append([event.y,event.x])
        flag2 = 0


root = Tk()

img = PhotoImage(file='Imagens/ian.pgm')
img2 = PhotoImage(file='Imagens/b.pgm')

posicao = StringVar()


p = Canvas(root, width = img.width()+15, height = img.height()+15)
p.place(x=20,y=20)
p.bind("<Button-1>", Posicao)
p.create_image(1, 1, anchor=NW, image=img)

posicao2 = StringVar()

p2 = Canvas(root, width = img.width()+15, height = img.height()+15)
p2.place(x=img.width()+20+20,y=20)
p2.bind("<Button-1>", Posicao2)
p2.create_image(1, 1, anchor=NW, image=img2)

center_window(img.width()*2+60, img.height()+200)

but = MyButton(p, 0-2, 0-2, 1)
vet.append([0,0])

but = MyButton(p, img.width()-1-2, 0-2, 1)
vet.append([0, img.width()-1])

but = MyButton(p, img.width()-1-2, img.height()-1-2, 1)
vet.append([img.height()-1, img.width()-1])

but = MyButton(p, 0-2, img.height()-1-2, 1)
vet.append([img.height()-1, 0])

but = MyButton(p2, 0-2, 0-2, 2)
vet2.append([0,0])

but = MyButton(p2, img.width()-1-2, 0-2, 2)
vet2.append([0, img.width()-1])

but = MyButton(p2, img.width()-1-2, img.height()-1-2, 2)
vet2.append([img.height()-1, img.width()-1])

but = MyButton(p2, 0-2, img.height()-1-2, 2)
vet2.append([img.height()-1, 0])


Button(root, text = 'Imprimir Tudo', command = ImprimeTudo).place(x=20, y=img.height()+30)
Button(root, text = 'Imprimir Pontos da Imagem Inicial', command = ImprimeIni).place(x=20, y=img.height()+120)
Button(root, text = 'Imprimir Pontos da Imagem Final', command = ImprimeFin).place(x=20, y=img.height()+60)
Button(root, text = 'Imprimir Poligonos', command = ImprimeLiga).place(x=20, y=img.height()+90)
Button(root, text = 'Visualizar Triangulacao', command = Visu).place(x=350, y=img.height()+90)
blq = Button(root, text = "Bloquear Criacao de Pontos", command = mudaFlag)
blq.place(x=350,y=img.height()+60)

root.mainloop()
