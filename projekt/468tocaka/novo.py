import cv2
import mediapipe as mp
import time

from mediapipe.python.solutions.face_mesh import FaceMesh

mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture('video1.mp4')
pTime = 0

mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
while True:      
    succuess, img = cap.read()
    dim=(int((img.shape[1])/4),int((img.shape[0])/4))
    img = cv2.resize(img,dim,fx=0,fy=0, interpolation=cv2.INTER_AREA)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACE_CONNECTIONS, drawing_spec, drawing_spec)

            
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                print(id, x,y)

            
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),3)
    cv2.imshow('image', img)
    cv2.waitKey(1)