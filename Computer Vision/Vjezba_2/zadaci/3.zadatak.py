import numpy as np
import cv2

capture = cv2.VideoCapture('red_car_moving.avi')

kernel = np.ones((7,7), np.uint8)
backSub = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = capture.read()

    if frame is None:
        break

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_fil = cv2.morphologyEx(backSub.apply(frame), cv2.MORPH_OPEN, kernel)
    
    cv2.imshow('Fram', frame)
    cv2.imshow('Frame', frame_fil)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()