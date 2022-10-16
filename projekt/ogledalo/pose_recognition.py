import mediapipe as mp
import cv2 

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#sa kamere slika
cap = cv2.VideoCapture(0)

#inicilijizacija holistic modela
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        #druga boja
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #detekcija
        results = holistic.process(image)
        print(results)

        #nazad u bgr
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        #crtanje lica
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

        #crtanje desne ruke
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        #crtanje lijeve ruke
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        #crtanje poze
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow('Holistic model', image)

        if cv2.waitKey(10) & 0xFF == ('q'):
            break



cap.release()
cv2.destroyAllWindows()
#video
'''cap = cv2.VideoCapture()
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Holistic model', frame)

    if cv2.waitKey(10) & 0xFF == ('q'):
        break

cap.release()
cv2.destroyAllWindows()'''