import cv2
import numpy as np

slika = cv2.imread('kovanice.png')
print(slika.shape)
grayscale_slika = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
grayscale_slika_blurred = cv2.GaussianBlur(grayscale_slika, (7, 7), 0)
circles = cv2.HoughCircles(grayscale_slika_blurred, cv2.HOUGH_GRADIENT, 1, 1, param1=170, param2=90)

circles = np.uint16(np.around(circles))

print(circles)