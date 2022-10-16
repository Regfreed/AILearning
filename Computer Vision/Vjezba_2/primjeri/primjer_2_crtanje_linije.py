import cv2
import numpy as np

slika = np.zeros((500, 500, 3))

pt1 = (0, 0)
pt2 = (250, 250)
line_color = [0, 255, 0]

cv2.line(slika, pt1, pt2, line_color)

cv2.imshow('linija', slika)
cv2.waitKey()
cv2.destroyAllWindows()
