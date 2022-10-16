import cv2

slika = cv2.imread("kamera1.jpg", cv2.IMREAD_COLOR)
slika[70, 210] = [0, 0, 0]
cv2.imshow("Slika s kamere", slika)
cv2.waitKey()
cv2.destroyAllWindows()

slika[:, :, 0] = 0
slika[:, :, 1] = 0
cv2.imshow("Slika s kamere samo s crvenom bojom", slika)
cv2.waitKey()
cv2.destroyAllWindows()
