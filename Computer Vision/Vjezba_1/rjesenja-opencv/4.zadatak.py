import cv2

slika = cv2.imread("kamera2.jpg")
slika = cv2.cvtColor(slika, cv2.COLOR_BGR2HSV)
for i in range(143,154):
    for j in range(274,279):
        slika[j,i] = [30,255,255]
slika = cv2.cvtColor(slika, cv2.COLOR_HSV2BGR)
cv2.imshow("slikica", slika)
print(slika.shape)
cv2.waitKey()
cv2.destroyAllWindows()