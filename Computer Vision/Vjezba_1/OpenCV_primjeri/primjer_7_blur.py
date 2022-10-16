import cv2

slika = cv2.imread('kamera1.jpg')
zamucena_slika = cv2.blur(slika, (5, 5))

cv2.imshow("Originalna slika", slika)
cv2.imshow("Zamucena slika", zamucena_slika)
cv2.waitKey()
cv2.destroyAllWindows()
