import cv2
import numpy as np

slika = np.zeros((500, 500, 3))

circle_center = (100, 100)
circle_radius = 100
circle_color = [0, 0, 255]

cv2.circle(slika, circle_center, circle_radius, circle_color)

cv2.imshow('kruznica', slika)
cv2.waitKey()
cv2.destroyAllWindows()
