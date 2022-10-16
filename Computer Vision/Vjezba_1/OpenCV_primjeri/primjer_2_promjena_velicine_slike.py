import cv2

slika = cv2.imread("kamera1.jpg", cv2.IMREAD_COLOR)
slika_s_novom_velicinom = cv2.resize(slika, None, fx=2.0, fy=2.0)

cv2.imshow("Originalna slika", slika)
cv2.imshow("Slika s novom velicinom", slika_s_novom_velicinom)
cv2.waitKey()
cv2.destroyAllWindows()
