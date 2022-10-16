import cv2

#capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture('car-overhead-1-hires.avi')

if not capture.isOpened():
    print('Unable to open: ' + 'car-overhead-1-hires.avi')
    exit(0)

while True:
    ret, frame = capture.read()

    if frame is None:
        break

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()