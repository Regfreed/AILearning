import cv2

slika = cv2.imread("kamera2.jpg", cv2.IMREAD_GRAYSCALE)

cv2.imshow("slikica", slika)
cv2.imwrite('kamera2_crno_bijela.jpg',slika)
print(slika.shape)
cv2.waitKey()
cv2.destroyAllWindows()