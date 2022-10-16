import cv2
import mediapipe as mp
import numpy as np
from numpy.lib.function_base import iterable
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# For static images:
IMAGE_FILES = ['crop.jpeg']
IMAGE_DATA = []
with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=1,
                            min_detection_confidence=0.7,
                            model_name='Shoe') as objectron:
  for idx, file in enumerate(IMAGE_FILES):
    image = cv2.imread(file)
    # Convert the BGR image to RGB and process it with MediaPipe Objectron.
    results = objectron.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw box landmarks.
    if not results.detected_objects:
      print(f'No box landmarks detected on {file}')
      continue
    print(f'Box landmarks of {file}:')
    annotated_image = image.copy()
    for detected_object in results.detected_objects:
      mp_drawing.draw_landmarks(
          annotated_image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
      mp_drawing.draw_axis(annotated_image, detected_object.rotation,
                           detected_object.translation)
      marks=detected_object.landmarks_2d
      IMAGE_DATA.append(detected_object)

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    canny = cv2.Canny(blur, 120, 255, 1)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    #find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    ROI = image[10:0+281,10:0+511]
    cv2.imwrite('ROI_{}.png'.format(1000), ROI)

    image_number = 0
    for c in cnts: 
      x,y,w,h = cv2.boundingRect(c)
      print(x,y,w,h)
      cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
      ROI = image[y:0+281, x:0+511]
     # ROI = image[y:y+437,y:y+513]
      cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
      image_number +=1

    cv2.imshow('thresh', thresh)
    cv2.imshow('canny', canny)
    cv2.imshow('original', image)
    cv2.imshow('ROI', ROI)
    cv2.imwrite(str(idx) + '.png', annotated_image)
    cv2.imshow('MediaPipe pic', annotated_image)
    if cv2.waitKey(5) & 0xFF == 27:
      break

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=1,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.99,
                            model_name='Shoe') as objectron:
  while cap.isOpened():
    success, image = cap.read()
    image2 = cv2.imread('The-Sydney-Harbor-Bridge.jpg')
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # pretvorba iz BGR u RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = objectron.process(image)

    # Draw the box landmarks on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(image2, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image2, detected_object.rotation,
                                 detected_object.translation)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)
    cv2.imshow('MediaPipe Objectron', image2)
    cv2.imshow('MediaPipe', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
print(IMAGE_DATA)
cap.release()