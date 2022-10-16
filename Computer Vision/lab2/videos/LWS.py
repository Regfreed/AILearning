import numpy as np
import cv2

print("Lane departure warning system")

video = cv2.VideoCapture('video2.mp4')
    

while(not video.isOpened()):
    video = cv2.VideoCapture('video1.mp4')
    print('ajd ponovo')

while True:
    ret, frame = video.read()

    if frame is None:
        break

    h,w,l=frame.shape

    roi_frame = frame[h//2:680,:]
    roi_frame=roi_frame[h//8:,:]
    kernel = np.ones((3,3), np.uint8)
    roi_frame = cv2.morphologyEx(roi_frame, cv2.MORPH_OPEN, kernel)
    
    gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    hsv_roi=cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100], dtype = 'uint8')
    upper_yellow = np.array([30, 255, 255], dtype='uint8' )

    mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray_roi, 150, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    #cv2.imshow('maska',mask_yw)
    mask_yw_image = cv2.bitwise_and(gray_roi, mask_yw)

    gauss_roi = cv2.GaussianBlur(mask_yw_image,(3,3),0)
    low_threshold = 50
    high_threshold = 150
    canny_roi = cv2.Canny(gauss_roi,low_threshold,high_threshold)
    #cv2.imshow('roi',canny_roi)
    linije = cv2.HoughLines(canny_roi,1,np.pi/180,45)
    print(linije)
    if linije is None:
        continue
    for linija in linije:
        for rho,theta in linija:
            print(rho,theta)
            if (theta<1 and theta>0.5) or (theta>2 and theta<2.5):         
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                
                
                if (theta<1.1 and theta>0.5):
                    pt1=(x1,y1)
                    pt2=(x2,y2)

                if (theta>2 and theta<2.5):
                    pt3=(x1,y1)
                    pt4=(x2,y2)
                
                cv2.line(roi_frame,(x1,y1),(x2,y2),(255,0,0),2)

    cv2.fillPoly(roi_frame, np.array([[pt1,pt2,pt3,pt4]]), (0,255,0), lineType=cv2.LINE_AA)



    cv2.imshow('roi',roi_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()