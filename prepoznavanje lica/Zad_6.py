import numpy as np
from imutils import face_utils
import imutils
import dlib
import cv2

def rect_to_bb(rect):

	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	return (x, y, w, h)


def shape_to_np(shape, dtype="int"):

	coords = np.zeros((68, 2), dtype=dtype)
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	return coords


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(1)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)   


while True:

	ret, image = cap.read()
	image = imutils.resize(image, width=300)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 1)
	for (i, rect) in enumerate(rects):
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		(x, y, w, h) = face_utils.rect_to_bb(rect)

		j = 0
		for (x, y) in shape:
			cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
			j+=1
			if(j==37):
				p1=(x,y)
			elif(j==38):
				p2=(x,y)
			elif(j==39):
				p3=(x,y)
			elif(j==40):
				p4=(x,y)
			elif(j==41):
				p5=(x,y)
			elif(j==42):
				p6=(x,y)
			if(j==43):
				p7=(x,y)
			elif(j==44):
				p8=(x,y)
			elif(j==45):
				p9=(x,y)
			elif(j==46):
				p10=(x,y)
			elif(j==47):
				p11=(x,y)
			elif(j==48):
				p12=(x,y)
		pa = np.sqrt(pow(p2[0]-p6[0],2) + pow(p2[1]-p6[1],2))
		pb = np.sqrt(pow(p3[0]-p5[0],2) + pow(p3[1]-p5[1],2))
		pc = np.sqrt(pow(p1[0]-p4[0],2) + pow(p1[1]-p4[1],2))
		EARr = (pa + pb)/(2*pc)
		
		pa = np.sqrt(pow(p8[0]-p12[0],2) + pow(p8[1]-p12[1],2))
		pb = np.sqrt(pow(p9[0]-p11[0],2) + pow(p9[1]-p11[1],2))
		pc = np.sqrt(pow(p7[0]-p10[0],2) + pow(p7[1]-p10[1],2))
		EARl = (pa + pb)/(2*pc)
	print(EARr)
	print(EARl)
	if(EARr<0.2 and EARl<0.2):
		print('otvori oci')
		
	cv2.imshow("Frame", image)
	
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

cap.release()
cv2.destroyAllWindows()
