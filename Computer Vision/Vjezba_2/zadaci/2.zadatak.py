import cv2
import numpy as np


slika = cv2.imread('kamera4.png')
slika_gry = cv2.cvtColor(slika,cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(slika_gry, 200, 255)
#lines = cv2.HoughLines(canny, 1, np.pi / 180, 200)
znak = cv2.imread('kamera4.png')
circles = cv2.HoughCircles(slika_gry,cv2.HOUGH_GRADIENT,1,1, param1=200, param2=90)
circles = np.uint16(np.around(circles))
print(circles)
for i in circles:
    for j in i:
        cv2.circle(znak,(j[0],j[1]),j[2],[255,0,0],2)

'''for i in lines:
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
    cv2.line(znak,pt1,pt2,[0,255,0],1)'''

cv2.imshow('original',slika)
cv2.imshow('znak',znak)
cv2.waitKey()
cv2.destroyAllWindows()