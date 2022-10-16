from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import torch
from torchvision.transforms.transforms import ToTensor
import gc

'''torch.cuda.empty_cache()
gc.collect()
'''

class MREZA(nn.Module):
    def __init__(self):
        super(MREZA, self).__init__()

        self.conv1 = nn.Conv2d(1,64,kernel_size=(3,3),padding=1)
        self.conv11 = nn.Conv2d(64,64,kernel_size=(3,3),padding=1)

        self.conv2 = nn.Conv2d(64,128,kernel_size=(3,3),padding=1)
        self.conv22 = nn.Conv2d(128,128,kernel_size=(3,3),padding=1)
        
        self.conv3 = nn.Conv2d(128,256,kernel_size=(3,3),padding=1)
        self.conv33 = nn.Conv2d(256,256,kernel_size=(3,3),padding=1)
        
        self.conv4 = nn.Conv2d(256,512,kernel_size=(3,3),padding=1)
        self.conv44 = nn.Conv2d(512,512,kernel_size=(3,3),padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.linear1 = nn.Linear(512*7*7,4096)
        self.linear2 = nn.Linear(4096,4096)
        self.linear3 = nn.Linear(4096,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv11(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv22(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv33(x)
        x = F.relu(x)
        x = self.conv33(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv44(x)
        x = F.relu(x)
        x = self.conv44(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv44(x)
        x = F.relu(x)
        x = self.conv44(x)
        x = F.relu(x)
        x = self.conv44(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 512*7*7)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.softmax(x, dim=1)

        return x


#parametri
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-3
batch_size = 16
epochs = 5
preprocess = transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224))])

mnist_training_dataset = MNIST('./data',train=True, download=True, transform=preprocess)
mnist_test_dataset =MNIST('./data', False, download=True, transform=preprocess)

mnist_training_dataloader = DataLoader(mnist_training_dataset, batch_size=batch_size, shuffle=True)
mnist_test_dataloader = DataLoader(mnist_test_dataset, batch_size=batch_size, shuffle=True)

mojModel = MREZA().to(device)
#mojModel.eval()

loss_fn = nn.CrossEntropyLoss()#funkcija koja nam govori koliko su dobri nasi parametri na izlazu svake epohe
optimizer = torch.optim.SGD(mojModel.parameters(), learning_rate)



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
    train(mnist_training_dataloader, mojModel, loss_fn, optimizer)
    test(mnist_test_dataloader, mojModel, loss_fn)

torch.save(mojModel, 'modelNovi.pt')
print('done')