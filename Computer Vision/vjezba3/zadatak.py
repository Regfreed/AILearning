import numpy as np
import cv2

MIN_MATCH_COUNT = 5 

img1 = cv2.imread('60_speed_limit.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('kamera4.png', cv2.IMREAD_GRAYSCALE)

#inicijalizacija SIFT

sift = cv2.SIFT_create()

#trazenje klkučnih točaka i računanje deskriptora pomoću SIFT metode
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

#ključne točke na slikama
img1_key=cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_key=cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)
#ukoliko 2 najbliža deskriptora nisu dovoljno različita, odbaci ih
good = []
for m,n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    M, _ = cv2.findHomography(src_pts,dst_pts, cv2.RANSAC, 5.0)
    h,w = img1.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1], [w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    img2 = cv2.polylines(img2,[np.int32(dst)], True,255,1, cv2.LINE_AA)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

else:
    print('nema dovoljno uparenih deskriptora - {}/{}'.format(len(good), MIN_MATCH_COUNT))
    matchesMask = None
cv2.imshow('parovi', img3)
cv2.imshow('detection', img2)
cv2.imshow('točke na prvoj', img1_key)
cv2.imshow('točke na drugoj', img2_key)
cv2.waitKey()
cv2.destroyAllWindows()