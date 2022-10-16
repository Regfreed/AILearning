import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import csv
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def find_csv_file(files):
    for file in files:
        if file.endswith('csv'):
            return file

def get_image_class(image_name, csv_file_name):
    csv_file = csv.reader(open(csv_file_name), delimiter=';')

    for row in csv_file:
        if row[0] == image_name:
            image_class = row[7]

            return int(image_class)

class TrafficSignsDataset(Dataset):
    def __init__(self, transforms=None):
        #ucitavanje podataka
        self.transforms = transforms
        self.image_names = []
        self.targets = []
        self.load_training_dataset()
        pass
    def __getitem__(self, index):
        #dohvačanje i-tog uzorka iz liste koje su kreirane u konstruktoru
        #funkcija treba vracati sliku i njenu oznaku
        image_name = self.image_names[index]
        target = self.targets[index]

        image = Image.open(image_name)

        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, target

    def __len__(self):
        #vraca velicinu skupa podataka
        return len(self.targets)
    
    def load_training_dataset(self):
        images_path = 'train'

        for dir in os.listdir(images_path):
            sub_directory = os.path.join(images_path, dir)
            if os.path.isdir(os.path.join(sub_directory)):
                files = os.listdir(sub_directory)
                csv_file_name = find_csv_file(files)
                csv_file_path = os.path.join(sub_directory, csv_file_name)
                for file_name in files:
                    if file_name.endswith('.ppm'):
                        images_path = os.path.join(sub_directory, file_name)
                        self.image_names.append(images_path)

                        image_class = get_image_class(file_name, csv_file_path)
                        self.targets.append(image_class)

preprocess = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
batch_size = 1

data_training_dataset = TrafficSignsDataset(preprocess)
#data_test_dataset =TrafficSignsDataset('./data', False, download=True, transform=preprocess)

data_training_dataloader = DataLoader(data_training_dataset, batch_size=batch_size, shuffle=True)
#data_test_dataloader = DataLoader(data_test_dataset, batch_size=batch_size, shuffle=True)

for batch, (X,y) in enumerate(data_training_dataloader):
    X = torch.squeeze(X)
    X = X.transpose(0,2)
    X = X.transpose(0,1)
    plt.imshow(X)
    plt.show()
    print(X.shape)
    break


dataset = TrafficSignsDataset()
