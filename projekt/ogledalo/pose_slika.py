import mediapipe as mp
import cv2 

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

image = cv2.imread('slika1.jpg')

with mp_holistic.Holistic(min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
    results = holistic.process(image)
    print(results)

    #crtanje lica
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

    #crtanje desne ruke
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    #crtanje lijeve ruke
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    #crtanje poze
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

cv2.imshow('Holistic model', image)
cv2.waitKey(0)
cv2.imwrite('slika22.jpg', image)
cv2.destroyAllWindows()




