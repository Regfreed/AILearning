'''
Lane departure warning system based on computer vision

Author: R.Grbic
'''

import numpy as np
import cv2
import math


def birdsEyeView(img, src, dst):
    h,w = img.shape[:2]
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)

    return warped, M, Minv


def detectEdges(image):    
    height, width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray_image, (5,5),0)
    canny_image = cv2.Canny(blurred_img, 100, 120)
    
    return canny_image


def filterByColor(image):
    imgHLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([180, 255, 255])
    white_mask = cv2.inRange(imgHLS, lower, upper)

    lower = np.uint8([ 20,   0, 100])
    upper = np.uint8([ 30, 255, 255])
    yellow_mask = cv2.inRange(imgHLS, lower, upper)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = cv2.bitwise_and(image, image, mask = mask)
    
    return result, yellow_mask, white_mask, mask


def drawLane(linesLeft, linesRight, frameToDraw):

    ymin = 0
    ymax = frameToDraw.shape[0]

    if linesLeft and linesRight:
        
        if linesLeft[0][1] != np.inf and linesLeft[0][1] != np.inf:

            x1_1 = int((ymin - linesLeft[0][1]) / linesLeft[0][0])
            x1_2 = int((ymax - linesLeft[0][1]) / linesLeft[0][0])
        else:
            x1_1 = linesLeft[0][3]
            x1_2 = linesLeft[0][3]
        
        if linesRight[0][1] != np.inf and linesRight[0][1] != np.inf:    
            
            x2_1 = int((ymin - linesRight[0][1]) / linesRight[0][0])
            x2_2 = int((ymax - linesRight[0][1]) / linesRight[0][0])
        else:
            x2_1 = linesRight[0][3]
            x2_2 = linesRight[0][3]


        if linesLeft[0][2] == 0 and linesRight[0][2] == 0:
            contours = np.array([[x1_1,ymin+RoIymin], [x2_1,ymin+RoIymin], [x2_2, ymax+RoIymin], [x1_2,ymax+RoIymin]])
            overlay = frameToDraw.copy()

            cv2.fillPoly(overlay, [contours], color=(0, 255, 100))
            cv2.addWeighted(overlay, 0.35, frameToDraw, 1 - 0.35, 0, frameToDraw)

    
    if linesLeft:
        if linesLeft[0][2] == 1:
            cv2.putText(frameToDraw, "Upozorenje!", (int(width/2)-140, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_4)
    
    if linesRight:
        if linesRight[0][2] == 1:
            cv2.putText(frameToDraw, "Upozorenje!", (int(width/2)-140, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_4)
        
    return frameToDraw
       


def findLines(img):

    lines = cv2.HoughLinesP(img, 1, np.pi/180, 15, minLineLength=10, maxLineGap=200)
    LinesInside = []
    linesLeft = []
    linesRight = []
    
    try:
        for line in lines:
            
            x1, y1, x2, y2 = line[0]
            if abs(x2-x1) <= 1.0:
                b = np.inf
                a = np.inf
                x_val = x1
                lineAngle = 90.0
            else:
                a = (y2-y1)/(x2-x1)
                b = y1 - a*x1
                x_val = (img.shape[0] - b)/a
                lineAngle = math.atan2((y2-y1), (x2-x1)) * 180/np.pi
            
            if x_val > 150.0 and x_val < 1200.0:

                # lijeva i desna linija
                if lineAngle > 10.0 and  lineAngle <=90.0:
                    if x_val > 450.0 and x_val < 800.0:
                        linesRight.append([a,b,1,x_val])
                    else:
                        linesRight.append([a,b,0,x_val])
                elif lineAngle < -10.0 and lineAngle >= -90.0:
                    if x_val > 450.0 and x_val < 800.0:
                        linesLeft.append([a,b,1,x_val])
                    else:
                        linesLeft.append([a,b,0,x_val])
    except:
        linesRight = []
        linesLeft = []

    return linesRight, linesLeft


def putInfoImg(img, text, loc):

    cv2.putText(img, 
                text, 
                (loc[0], loc[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)


def imgFiltering(img):
    height, width = img.shape[:2]
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray_image, (5,5),0)
    equ = cv2.equalizeHist(blurred_img)
    canny_image = cv2.Canny(equ, 100, 120)

    return canny_image




pathResults = 'results/'
pathVideos = 'videos/'
videoName  = 'video1.mp4'

cap = cv2.VideoCapture(videoName)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

cv2.namedWindow('Input image',cv2.WINDOW_NORMAL)
cv2.namedWindow('RoI Edges',cv2.WINDOW_NORMAL)
cv2.namedWindow('Filtered RoI',cv2.WINDOW_NORMAL)
cv2.namedWindow('Yellow',cv2.WINDOW_NORMAL)
cv2.namedWindow('White',cv2.WINDOW_NORMAL)
cv2.namedWindow('Lines',cv2.WINDOW_NORMAL)

k = 0
frameToSave = 420

RoIymin = 460
RoIymax = 620

kernel = np.ones((5,5),np.uint8)
time = 1

while(True):
    e1 = cv2.getTickCount()

    # ucitaj frame
    ret, frame = cap.read()
    frame_cpy = frame.copy()
    if ret == False:
        print("Video end!")
        break
    else:
        k = k + 1

    # regija od interesa
    frameRoI = frame[RoIymin:RoIymin+(RoIymax-RoIymin),0:width,:]
    
    # filtriranje po boji
    RoIFiltered, yellow_mask, white_mask, mask = filterByColor(frameRoI)

    # detekcija rubova
    RoIEdges = detectEdges(RoIFiltered)

    # pronadji lijevu i desnu liniju
    linesRight, linesLeft = findLines(RoIEdges)

    # prikazi linije
    RoILines = drawLane(linesLeft, linesRight, frame)

    # prikazi rezultate
    putInfoImg(frame, "frame: " + str(k), ((50,50)))
    putInfoImg(frame, "FPS: " + str(int(1/time)), ((50,100)))

    cv2.imshow('Input image',frame)
    cv2.imshow('RoI Edges',RoIEdges)
    cv2.imshow('Yellow', yellow_mask)
    cv2.imshow('White', white_mask)
    cv2.imshow('Lines', RoILines)


    if k == frameToSave:
        cv2.imwrite("frame_%d.jpg" % k, RoILines)   
    
    key =  cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break

    if key == ord('p'):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            if key2 == ord('p'):
                break
            if key2 == ord('q'):
                break
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    #print(1.0/time * 1000)


cap.release()
cv2.destroyAllWindows()