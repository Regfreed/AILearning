import cv2

slika = cv2.imread("kamera1.jpg", cv2.IMREAD_COLOR)

cv2.imshow('Slika s kamere', slika)
cv2.waitKey()
cv2.destroyAllWindows()
