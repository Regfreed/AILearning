import cv2
import os
import re
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 


#  Felzenszwalb et al.
def non_max_suppression_slow(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list, add the index
		# value to the list of picked indexes, then initialize
		# the suppression list (i.e. indexes that will be deleted)
		# using the last index
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		suppress = [last]
        # loop over all indexes in the indexes list
		for pos in xrange(0, last):
			# grab the current index
			j = idxs[pos]
			# find the largest (x, y) coordinates for the start of
			# the bounding box and the smallest (x, y) coordinates
			# for the end of the bounding box
			xx1 = max(x1[i], x1[j])
			yy1 = max(y1[i], y1[j])
			xx2 = min(x2[i], x2[j])
			yy2 = min(y2[i], y2[j])
			# compute the width and height of the bounding box
			w = max(0, xx2 - xx1 + 1)
			h = max(0, yy2 - yy1 + 1)
			# compute the ratio of overlap between the computed
			# bounding box and the bounding box in the area list
			overlap = float(w * h) / area[j]
			# if there is sufficient overlap, suppress the
			# current bounding box
			if overlap > overlapThresh:
				suppress.append(pos)
		# delete all indexes from the index list that are in the
		# suppression list
		idxs = np.delete(idxs, suppress)
	# return only the bounding boxes that were picked
	return boxes[pick]

dataset_path = 'dataset/'
mean_pedestrian_width = 101
mean_pedestrian_height = 264

def load_ped_dataset():    #dohvačanje slika iz ped  foldera i pravljenja HOG-a
    ped_images_path = os.path.join(dataset_path, 'ped')

    pedestrian_hogs = []

    for file in os.listdir(ped_images_path):  #vrača popis svih fileova iz foldera
        if file.endswith('.png'):              #provjera dali je .png file a ne slucajno mozda .txt
            image_name = os.path.join(ped_images_path, file)

            # Učitaj sliku u grayscale formatu
            image_gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)

            # Promijeni veličinu slike na (mean_pedestrian_width, mean_pedestrian_height)
            image_gray = cv2.resize(image_gray,(mean_pedestrian_width,mean_pedestrian_height ))

            # Izračunaj HOG značajke pomoću hog() funkcije
            fd, hog_slika = hog(image_gray, orientations=8, visualize=True)  #fd je jedan vektor brojeva svake čelije iz HOG-a
        

            # Spremi izračunate HOG značajke u listu
            pedestrian_hogs.append(fd)


    # Pretvori listu HOG značajki u Numpy polje
    pedestrian_hogs = np.array(pedestrian_hogs)

    return pedestrian_hogs

def load_no_ped_dataset():
    no_ped_images_path = os.path.join(dataset_path,'no_ped')

    no_pedestrian_hog = []

    for file in os.listdir(no_ped_images_path):
        if file.endswith('.png'):
            image_name = os.path.join(no_ped_images_path, file)

            image_gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
            image_gray = cv2.resize(image_gray,(mean_pedestrian_width,mean_pedestrian_height))

            fd, hog_image = hog(image_gray, orientations=8, visualize=True)

            no_pedestrian_hog.append(fd)

    no_pedestrian_hog = np.array(no_pedestrian_hog)

    return no_pedestrian_hog

ped_dataset = load_ped_dataset()
no_ped_dataset = load_no_ped_dataset()

# za gornje vrijednosti napravitti izlaze(1-pjesak, 0- nije pjesak) mora biti dimenzije(broj elementa X 1)
ped_out = np.ones(ped_dataset.shape[0])
no_ped_out = np.zeros(no_ped_dataset.shape[0])

# spojiti x-eve u dataset_in listu i y-one u dataset_out listu
dataset_in = np.concatenate((ped_dataset,no_ped_dataset))
dataset_out = np.concatenate((ped_out,no_ped_out))

#podjela ulaza i izlaza u train i test podataka pomoću funkcije train_test_split()
x_train, x_test, y_train, y_test = train_test_split(dataset_in,dataset_out, test_size=0.2)

clf = MLPClassifier(max_iter=300).fit(x_train, y_train) #pomoću fit() mi treniramo mrežu

print(clf.score(x_test,y_test))

#idemo učitati sliku za testiranje

test_slika_path = os.path.join(dataset_path,'morning-walk.png')
test_slika = cv2.imread(test_slika_path)
test_slika_gray = cv2.cvtColor(test_slika, cv2.COLOR_BGR2GRAY)

PROBA_MAX = 0.8
cordinates =[]

track_window =(0,mean_pedestrian_height,mean_pedestrian_width,mean_pedestrian_height)
for i in range(0,test_slika_gray.shape[0]-mean_pedestrian_height, 5):
    for j in range(0,test_slika_gray.shape[1]-mean_pedestrian_width, 5):

        roi = test_slika_gray[i:mean_pedestrian_height + i, j: j+mean_pedestrian_width]

        fd_test_slike = hog(roi, orientations=8)

        fd = fd_test_slike.reshape((1,-1))


        proba = clf.predict_proba(fd)# clf.predict_proba() prima 4 parametra pa treba 1D array ili ti listu vrijednosti vektora malo preoblikovat
        if PROBA_MAX<proba[0][1]:
            #PROBA_MAX=proba[0][1]
            cordinates.append((j,i))
            test_slika = cv2.rectangle(test_slika,(j,i), (j+mean_pedestrian_width,i+mean_pedestrian_height), 255,2)
print(cordinates)           
#test_slika = cv2.rectangle(test_slika,(x,y), (x+mean_pedestrian_width,y+mean_pedestrian_height), 255,2)

cv2.imshow('kraj', test_slika)
cv2.waitKey()
cv2.destroyAllWindows()
