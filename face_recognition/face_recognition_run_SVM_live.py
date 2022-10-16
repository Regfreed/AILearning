import numpy as np
import imutils
import pickle
import cv2

recognizerName = "recognizer.pickle"
personNames = "persons.pickle"

prototxt_path = 'models/deploy.prototxt'
caffemodel_path = 'models/res10_300x300_ssd_iter_140000.caffemodel'
detector = cv2.dnn.readNet(caffemodel_path, prototxt_path)

embedder_path = 'models/nn4.small2.v1.t7'
embedder = cv2.dnn.readNetFromTorch(embedder_path)

recognizer = pickle.loads(open(recognizerName, "rb").read())
le = pickle.loads(open(personNames, "rb").read())

confidenceTH = 0.7
confidenceRecognizeTH = 0.6

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),(104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()
    
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidenceTH:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue
            
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
            if proba > confidenceRecognizeTH:
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
            else:
                cv2.putText(frame, "unknown", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
