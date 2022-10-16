import numpy as np
import cv2

cap = cv2.VideoCapture("car_meanshift.mp4")

# dohvati prvu sliku iz videa
ret,frame = cap.read()

# postavi početnu lokaciju objekta kojeg želimo pratiti
x, y, w, h = 223, 202, 34, 20
track_window = (x, y, w, h)

# pripremi područje objekta za praćenje
roi = frame[y:y+h, x:x+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((170, 50, 50)), np.array((180, 255 ,255 )))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Postavi uvjet za kraj -> nakon 10 iteracija ili pomakni se za barem jedan piksel
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # primijeni meanshift kako bi dobio novu lokaciju objekta
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # prikaži novu lokaciju
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        cv2.imshow('img2',img2)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()