#konvolucijska neuronska mreza i MNIST
#prvo treba ucitati dataset za rad 
#Dataset- sadrzi uzorke i odgovarajuce oznake;  DataLoader-omogućava lakše iteriranje kroz Dataset. koristi se za dohvaćanje uzoraka prilikom treniranja
#torchvision.dataset-za skup podataka slika ; torchvision.datasets.iamgenet- 1000 klasa, 1.2M slika za treniranje
#torchvision.datasets.VOCSegmentation- koristi se za zadatak segmetancije, 20 klasa, 11530 slika za trening/validaciju

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.transforms import ToTensor  #sluzi za transformaciju slika na neki nacin koji mi odlucimo(npr. pretvori sliku u tensor (transform=toTensor), ili resize,...)
                                    #ili ako zelimo vise transformacija odjednom onda koristimo transform.Compose([transform.Resize((640,640)), transform.ToTensor()])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

mnist_training_dataset = MNIST('./data',train=True, download=True, transform=ToTensor())
mnist_test_dataset =MNIST('./data', False, download=True, transform=ToTensor())

mnist_training_dataloader = DataLoader(mnist_training_dataset, batch_size=1, shuffle=True)
mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=1, shuffle=True)
i=0
print('using: ' + device)
for train_images, train_labels in mnist_training_dataloader:
    i+=1
    img = train_images.squeeze()
    imgplot=plt.imshow(img)
    plt.title(train_labels.item())
    plt.show()
    if i>2:
        break

#definiranje neuronske mreže
#nasljeđivanje neke klase je OBAVEZNO



#zadatak: prikazati par trening slika i odgovarajuću klasu
#primjer dohvaćanja prve slike i klsae

#img, label = mnist_training_dataset[0]
#img = torch.sqeeze(img)

#plt.imshow(img)