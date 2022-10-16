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
ped_dataset = load_ped_dataset()
print(ped_dataset.shape)