import cv2

slika_bgr = cv2.imread("kamera1.jpg")
slika_hsv = cv2.cvtColor(slika_bgr, cv2.COLOR_BGR2HSV)
slika_hsv[70, 211] = [0, 255, 255]
slika_bgr_izmijenjena = cv2.cvtColor(slika_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow("izmijenjena slika", slika_bgr_izmijenjena)
cv2.waitKey()
cv2.destroyAllWindows()
