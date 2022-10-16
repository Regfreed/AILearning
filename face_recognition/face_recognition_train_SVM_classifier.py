from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle


dataset_dir = "myDataset/"
detections_dir = "train_detections/"
encodingsName = "train_encodings.pickle"
recognizerName = "recognizer.pickle"
personNames = "persons.pickle"

data = pickle.loads(open(encodingsName, "rb").read())
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(recognizerName, "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(personNames, "wb")
f.write(pickle.dumps(le))
f.close()