import cv2
import numpy as np

slika_bgr = cv2.imread('zeleni_semafor.jpg')
slika_hsv = cv2.cvtColor(slika_bgr, cv2.COLOR_BGR2HSV)

zelena_donja_granica = np.array([75, 80, 80])
zelena_gornja_granica = np.array([85, 255, 255])

maska = cv2.inRange(slika_hsv, zelena_donja_granica, zelena_gornja_granica)
filtrirana_slika_hsv = cv2.bitwise_and(slika_hsv, slika_hsv, mask=maska)
filtrirana_slika_bgr = cv2.cvtColor(filtrirana_slika_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("Originalna slika", slika_bgr)
cv2.waitKey()
cv2.imshow("Maska", maska)
cv2.waitKey()
cv2.imshow("Filtrirana slika", filtrirana_slika_bgr)
cv2.waitKey()
cv2.destroyAllWindows()
