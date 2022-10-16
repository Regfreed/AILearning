from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from network import CNN
from torchvision.transforms import ToTensor  #sluzi za transformaciju slika na neki nacin koji mi odlucimo(npr. pretvori sliku u tensor (transform=toTensor), ili resize,...)
                                    #ili ako zelimo vise transformacija odjednom onda koristimo transform.Compose([transform.Resize((640,640)), transform.ToTensor()])

#parametri za podešavanje treninga mreže
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
batch_size = 64
epochs = 5

mnist_training_dataset = MNIST('./data',train=True, download=True, transform=ToTensor())
mnist_test_dataset =MNIST('./data', False, download=True, transform=ToTensor())

mnist_training_dataloader = DataLoader(mnist_training_dataset, batch_size=1, shuffle=True)
mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=1, shuffle=True)

#ovdje treba loadat mrezu


MyCNN = CNN().to(device)  #prebacili smo svoju mrezu na GPU ako imamo ako ne bit ce na CPU

loss_fn = nn.CrossEntropyLoss()#funkcija koja nam govori koliko su dobri nasi parametri na izlazu svake epohe
optimizer = torch.optim.SGD(MyCNN.parameters(), learning_rate)

#definiranje trening metode
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print('[' + str(current) + '/' + str(size) + ']' + ' Loss: ' + str(loss))


#definiranje test metode za evaluaciju

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()                    #ova linija iskljućuje slojeve koji nisu potrebni tijekom korištenja modela(npr. Dropout)
    test_loss, correct = 0, 0
    with torch.no_grad():           #ova linija onemogućuje računanje gradijenata(koji se inače računaju pomoću backward() funkcije)
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print('Accuracy: ' + str(100 * correct) + '%, Average loss: ' + str(test_loss))

#petlja optimizacije. sada vršimo treniranje modela

for e in range(epochs):
    print('Epoch '+ str(e+1))
    train(mnist_training_dataloader, MyCNN, loss_fn, optimizer)
    test(mnist_test_dataloader, MyCNN, loss_fn)

torch.save(MyCNN, 'model.pt')
print('done')