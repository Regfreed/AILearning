import cv2
img_src = cv2.imread("primjer.png")
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
ret, img_gray_th = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

# detektiraj konture
cnts, hierarchy = cv2.findContours(image=img_gray_th, mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_NONE)

# prikazi konture na originalnoj slici
cv2.drawContours(image=img_src, contours=cnts, contourIdx=-1,color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
cv2.imwrite("rezultat.png", img_src)
cv2.imshow("rezultat", img_src)
cv2.waitKey(0)
cv2.destroyAllWindows()