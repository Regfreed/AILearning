from torch import nn
import torch.nn.functional as F


#definiranje neuronske mreže
#nasljeđivanje neke klase je OBAVEZNO (u ovom slucaju nn.Module)
#u init definiramo slojeve mreze
# u forward cemo pozivati slojeve
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1,10,kernel_size=(5,5),padding=2)
        self.conv2 = nn.Conv2d(10,20,kernel_size=(5,5),padding=2)

        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.linear1 = nn.Linear(980,50)
        self.linear2 = nn.Linear(50,10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 20*7*7)
        x = self.linear1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = F.softmax(x, dim=1)

        return x