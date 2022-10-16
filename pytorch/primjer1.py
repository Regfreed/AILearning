from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torch

from torchvision import transforms
from torchvision.transforms import ToTensor  #sluzi za transformaciju slika na neki nacin koji mi odlucimo(npr. pretvori sliku u tensor (transform=toTensor), ili resize,...)
                                    #ili ako zelimo vise transformacija odjednom onda koristimo transform.Compose([transform.Resize((640,640)), transform.ToTensor()])

mnist_training_dataset = MNIST('./data',train=True, download=True, transform=ToTensor())
mnist_test_dataset =MNIST('./data', False, download=True, transform=ToTensor())

mnist_training_dataloader = Da('./data')

#zadatak: prikazati par trening slika i odgovarajuću klasu
#primjer dohvaćanja prve slike i klsae

img, label = mnist_training_dataset[0]
img = torch.sqeeze(img)

plt.imshow(img)