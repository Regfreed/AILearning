from keras.models import load_model
from tensorflow import keras
import numpy as np
import cv2


# ucitaj checkpoint
model = load_model('checkpoints/model.23-0.13.h5')


prototxt_path = 'models/deploy.prototxt'
caffemodel_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNet(caffemodel_path, prototxt_path)

cap = cv2.VideoCapture(1)
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)  


while True:
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]
    frame_resized = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(frame_resized, 1.0, (300, 300), (104.0, 177.0, 123.0))
    
    net.setInput(blob)
    detections = net.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence < 0.7:
            continue
        
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = abs(box.astype("int"))
        if ((endX-startX) > 20) and ((endY -startY) > 20):
            img = cv2.resize(frame[startY:endY, startX:endX], (80,80))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255
            img = img.reshape(1,80,80,3)

            probab = model.predict(img)
            label = np.argmax(probab)
            #print(probab*100.0)
            if label == 0:            
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                cv2.putText(frame, "Stavite masku!",(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
 
cv2.destroyAllWindows()
cap.release()