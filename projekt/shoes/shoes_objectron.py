import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# For static images:
""" IMAGE_FILES = []
with mp_objectron.Objectron(static_image_mode=True,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
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
      cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
 """
# For webcam input:
cap = cv2.VideoCapture(0)
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=2,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.5,
                            model_name='Shoe') as objectron:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = objectron.process(image)

    # Draw the box landmarks on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    original = image.copy()
    if results.detected_objects:
        for detected_object in results.detected_objects:
            print(detected_object.landmarks_2d)
            mp_drawing.draw_landmarks(
              image, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
            mp_drawing.draw_axis(image, detected_object.rotation,
                                 detected_object.translation)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    canny = cv2.Canny(blur, 120, 255, 1)
    kernel = np.ones((5,5),np.uint8)
    dilate = cv2.dilate(canny, kernel, iterations=1)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    #find contours
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#pokusao si ovako napraviti pa neide sada moras probati predati u loop landmarke a ne konture i pokusat spremiti roi
    #Iterate thorugh contours Obtain bounding rectangle and extract ROI
    image_number = 0
    for c in cnts: 
      x,y,w,h = cv2.boundingRect(c)
      print(x,y,w,h)
      cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 2)
      ROI = original[y:y+h, x:x+w]
      cv2.imwrite('ROI_{}.png'.format(image_number), ROI)
      image_number +=1

    # Add alpha channel
    #b,g,r = cv2.split(ROI)
    #alpha = np.ones(b.shape, dtype=b.dtype) * 50
    #ROI = cv2.merge([b,g,r,alpha])

    cv2.imshow('thresh', thresh)
    cv2.imshow('canny', canny)
    cv2.imshow('original', original)
    cv2.imshow('ROI', ROI)
    cv2.imshow('MediaPipe Objectron', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()