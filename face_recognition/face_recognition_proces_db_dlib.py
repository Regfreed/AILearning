'''
Skripta procesira bazu lica koja se nalazi u folderu train. Na svaku sliku u train folderu primjenjuje
se detektor lica te se izracunava feature vektor. Feature vektori se zajedno s imenom podfoldera spremaju u pickle na disk. 
'''

from imutils import paths
from imutils import face_utils
import face_recognition
import pickle
import cv2
import os

# prebacuje dlib format BB u oblik pogodan za OpenCV - samo za debug
def rect_to_bb(rect):
	x = rect[3]
	y = rect[0]
	w = rect[1] - x
	h = rect[2] - y

	return (x, y, w, h)

# relativne putanje
dataset_dir = "train/"
detections_dir = "train_detections/"
encodingsName = "train_encodings_dlib.pickle"

# kreiraj listu s putanjama do svake slike u train direktoriju
imagePaths = list(paths.list_images(dataset_dir))

# inicijaliziraj liste za imena i feature
knownEncodings = []
knownNames = []

# procesiraj svaku sliku
for (i, imagePath) in enumerate(imagePaths):

	# izluci ime svake osobe iz imena podirektorija
	print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# ucitaj sliku pomocu OpenCV (BGR) format i prebaci u RGB format koji koristi dlib
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


	# detektiraj bounding box na slici; model moze biti cnn ili hog
	boxes = face_recognition.face_locations(rgb, model="cnn")

	# za debug - prikaz detektiranog lica
	#(x, y, w, h) = rect_to_bb(boxes[0])
	#cropped = image[y:y+w, x:x+h]
	#cv2.imwrite(os.path.join(detections_dir,name), cropped)
	#cv2.waitKey(0)

	# izracunaj feature za svako lice
	encodings = face_recognition.face_encodings(rgb, boxes)
	
	for encoding in encodings:
		
		#spremi feature i ime osobe
		knownEncodings.append(encoding)
		knownNames.append(name)

# spremi sve na disk u jedan file (pickle)
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encodingsName, "wb")
f.write(pickle.dumps(data))
f.close()