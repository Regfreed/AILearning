import cv2

slika = cv2.imread('kamera2.jpg')

grayscale = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
slika_zamucena = cv2.GaussianBlur(grayscale, (5, 5), 0)

sobel_x = cv2.Sobel(slika_zamucena, cv2.CV_64F, 1, 0, ksize=5)

sobel_x_abs = cv2.convertScaleAbs(sobel_x)

cv2.imshow("Originalna slika", slika)
cv2.imshow("Sobel x", sobel_x_abs)
cv2.waitKey()
cv2.destroyAllWindows()
