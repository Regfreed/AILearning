import cv2
import os
import re
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier 

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

test_slika_path = os.path.join(dataset_path,'test_img.png')
test_slika = cv2.imread(test_slika_path)
test_slika_gray = cv2.cvtColor(test_slika, cv2.COLOR_BGR2GRAY)

PROBA_MAX = 0
PROZOR = 0
x,y=0,0

track_window =(0,mean_pedestrian_height,mean_pedestrian_width,mean_pedestrian_height)
for i in range(0,test_slika_gray.shape[0]-mean_pedestrian_height, 5):
    for j in range(0,test_slika_gray.shape[1]-mean_pedestrian_width, 5):

        roi = test_slika_gray[i:mean_pedestrian_height + i, j: j+mean_pedestrian_width]

        fd_test_slike = hog(roi, orientations=8)

        fd = fd_test_slike.reshape((1,-1))


        proba = clf.predict_proba(fd)# clf.predict_proba() prima 4 parametra pa treba 1D array ili ti listu vrijednosti vektora malo preoblikovat
        if PROBA_MAX<proba[0][1]:
            PROBA_MAX=proba[0][1]
            x=j
            y=i
            
test_slika = cv2.rectangle(test_slika,(x,y), (x+mean_pedestrian_width,y+mean_pedestrian_height), 255,2)
cv2.imshow('kraj', test_slika)
cv2.waitKey()
cv2.destroyAllWindows()
