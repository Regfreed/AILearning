import cv2
import numpy as np

org_slika = cv2.imread("sudoku.jpg")
slika = cv2.cvtColor(org_slika, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(slika, 100, 200)

lines = cv2.HoughLines(canny, 1, np.pi / 180, 200)

print(lines)

cv2.imshow("sudoku slika", org_slika)
cv2.imshow("canny", canny)
cv2.waitKey()
cv2.destroyAllWindows()