import face_recognition
import pickle
import cv2

# ulazna slika
imagePath = "val/ben_afflek/04.jpg"

# baza
encodingsName = "train_encodings_dlib.pickle"

# ucitaj encodingse baze slika
data = pickle.loads(open(encodingsName, "rb").read())

# ucitaj ulaznu sliku
image = cv2.imread(imagePath)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# detektiraj lica u slici i izracunaj feature vektor za svako lice
boxes = face_recognition.face_locations(rgb, model="cnn")
encodings = face_recognition.face_encodings(rgb, boxes)

# inicijalizacija liste s imenima identificiranih osoba
names = []

for encoding in encodings:

	# feature vektor svake detekcije usporedi sa bazom
	matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.6)
	name = "Unknown"

    # ako se podudaraju featuri vektori (udaljenost manja od thesholda)
	if True in matches:

		# pronadji indekse s kojim licima je uparen trenutni feature vektor
		matchedIdxs = [i for (i, b) in enumerate(matches) if b]
		counts = {}

		# na temelju indeksa izbroji koliko puta je koja osoba uparena s trenutnim feature vektorom
		for i in matchedIdxs:
			name = data["names"][i]
			counts[name] = counts.get(name, 0) + 1
			
		# osoba s najvecim brojem je predikcija
		name = max(counts, key=counts.get)

	# osvjezi listu imena
	names.append(name)
    

# prikaz prepoznatih lica
for ((top, right, bottom, left), name) in zip(boxes, names):

	# prikazi bounding box i ime osobe
	cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
	y = top - 15 if top - 15 > 15 else top + 15
	cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

# prikazi krajnji rezultat
cv2.imshow("Image", image)
cv2.waitKey(0)
