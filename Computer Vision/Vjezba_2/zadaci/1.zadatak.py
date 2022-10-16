import cv2
import numpy as np

slika = cv2.imread('kamera3.jpeg')
slika_gry = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
ret, threshold_slika = cv2.threshold(slika_gry,175,255,cv2.THRESH_BINARY)
canny = cv2.Canny(slika_gry, 210, 220)
lines = cv2.HoughLines(canny, 1, np.pi / 180, 200)

for i in lines:
    print(i[:,1][0])
    a=np.cos(i[:,1][0])
    b=np.sin(i[:,1][0])
    x0=a*i[:,0][0]
    y0=b*i[:,0][0]
    x1=int(x0 + 1000*(-b))
    y1=int(y0 + 1000*(a))
    x2=int(x0 - 1000*(-b))
    y2=int(y0 - 1000*(a))
    pt1=(x1,y1)
    pt2=(x2,y2)
    cv2.line(slika,pt1,pt2,[0,255,0],2)

print(lines)
cv2.imshow('canny', canny)
cv2.imshow('original',slika)
cv2.imshow('binarna', threshold_slika)
cv2.waitKey()
cv2.destroyAllWindows()



