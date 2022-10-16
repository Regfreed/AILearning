'''Generirajte 100 slučajnih cjelobrojnih brojeva u rasponu [1, 10]. Prikažite histogram generiranih
brojeva.
Za generiranje cjelobrojnih brojeva možete koristiti Numpy funkciju np.random.randint().'''


import numpy as np
import matplotlib.pyplot as plt
import cv2

x = np.random.randint(1,11,50)
img = cv2.imread('raskrizje.png')
color = ('b','g','r')

plt.hist(x, 10)
plt.show()
for i, col in enumerate(color):
    hist = cv2.calcHist([img],[i],None, [256],[0,256] )
    plt.plot(hist, col)
    plt.xlim([0,256])

plt.show()
img_grey = cv2.imread('raskrizje.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow('slika', img_grey)
plt.hist(img_grey.ravel(),256,[0,256])
plt.show()