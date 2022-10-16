import cv2
import numpy as np

slika = cv2.imread('sum.png')

kernel = np.ones((5,5), np.uint8)
bez_suma = cv2.morphologyEx(slika, cv2.MORPH_OPEN, kernel)

cv2.imshow('Originalna', slika)
cv2.imshow('Bez suma', bez_suma)
cv2.waitKey()
cv2.destroyAllWindows()