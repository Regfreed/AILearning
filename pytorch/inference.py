#nasljedujemo istrenirani model i njemu predajemo onda novu sliku koju jos nikad nije vidio pa citamo rezultate kako je klasificirao tu sliku


from network import CNN
import torch
import cv2

myModel = torch.load('model.pt')
myModel.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img = cv2.imread('test_image.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = torch.tensor(img, dtype=torch.float)
img = torch.unsqueeze(img, 0)
img = torch.unsqueeze(img, 0)
img = img.to(device)

a = myModel(img)
print(img.shape)
print(a)