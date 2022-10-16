import cv2

slika = cv2.imread("kamera2.jpg")
for i in range(143,154):
    for j in range(274,279):
        slika[j,i] = [0, 255, 255]
cv2.imshow("slikica", slika)
print(slika.shape)
cv2.waitKey()
cv2.destroyAllWindows()