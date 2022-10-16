import cv2

slika = cv2.imread('kamera2.jpg')
canny = cv2.Canny(slika, 100, 200)

cv2.imshow("Originalna slika", slika)
cv2.imshow("Canny", canny)
cv2.waitKey()
cv2.destroyAllWindows()
