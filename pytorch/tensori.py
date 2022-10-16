import torch
import numpy as np

a = np.array([1,2,3])

a = torch.tensor(a)

print(a)

b=torch.ones_like(a, dtype=torch.float)

print(b)

#funkcije za pravljenje matrica samo sa nula, a ones radi samo sa jedinicama, rand radi sa float izmedu 0 i 1
x = torch.zeros(size=(2,3))
print(x)

#atributi
#shape- dimenzije tensora,  dtype- tip podataka,  device-di radi 

print(x.device)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

#prebacivanje sa cpu na gpu 
x = x.to(device)

print(x.device)
print(x.shape)
print(x.dtype)

b = torch.rand(2,3)

print(b)
print(b[0])

a = torch.tensor([[1,2],[3,4],[5,6]])
b = torch.tensor([[1,2,3],[4,5,6]])
c = a.matmul(b)
print(c)

#dohvacanje python broja ako tensor ima 1 element
y = torch.tensor([4])
z = y.item()
print(type(z))

#squeeze brise dimenziju ako je ona 1, a unsqueeze dodaje tu 1 dimenziju, recimo kad hocemo dodati minibatch

#promjena dimezije sa view
a = torch.rand(10,10)
print(a.shape)
b = a.view(100)
print(b.shape)
c = a.view(-1,20)
print(c.shape)

#kada zelimo naci mjesto koje ima najvecu vrijednost

a = torch.tensor([1,2,3,10,4,5])

print(a.argmax().item())