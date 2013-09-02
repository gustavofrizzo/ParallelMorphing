from adpil import *
from numpy import *

vimg = []

for i in range(10):
	vimg.append(adread('rrr'+str(i)+'.pgm'))
	
final = zeros([vimg[0].shape[0]*2, vimg[0].shape[1]*5])

l = 0 #linha
c = 0 #coluna
cuidal = 0
cuidac = 0 
cont = 0
for img in vimg:
	cont += 1
	l = cuidal
	c = cuidac
	print cont
	for i in range(img.shape[0]-1):
		for j in range(img.shape[1]-1):
			final[l,c] = img[i,j]
			c+=1
		l+=1
		c = cuidac
	if cont < 5:
		cuidal = 0
		cuidac = cuidac +  img.shape[1]
	if cont == 5:
		cuidal = img.shape[0]
		cuidac = 0
	if cont > 5:
		cuidal = img.shape[0]
		cuidac = cuidac +  img.shape[1]
adwrite('final3.png', final)
#adshow(final)
