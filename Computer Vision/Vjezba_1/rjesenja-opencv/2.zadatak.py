import cv2

slika = cv2.imread("kamera2.jpg")
nova = cv2.resize(slika,(1000,650))
cv2.imshow("slikica", nova)
print(nova.shape)
cv2.waitKey()
cv2.destroyAllWindows()