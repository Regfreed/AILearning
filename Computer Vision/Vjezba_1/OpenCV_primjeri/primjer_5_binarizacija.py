import cv2

slika = cv2.imread("kamera1.jpg", cv2.IMREAD_GRAYSCALE)
ret, threshold_slika = cv2.threshold(slika, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Originalna slika", slika)
cv2.imshow("Crno-bijela (binarna) slika", threshold_slika)
cv2.waitKey()
cv2.destroyAllWindows()
