import numpy as np
import kagglehub
from torch import nn
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from torchmetrics.classification import Accuracy, BinaryAccuracy
#path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
file_path = "./dataset/versions/1/PetImages"
#file_path = pathlib.Path(path) / 'PetImages'
#file_path_cat = pathlib.Path(path) / 'PetImages' / 'Cat'
#file_path_dog = pathlib.Path(path) / 'PetImages' / 'Dog'

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

transforms = transforms.Compose([
  v2.RGB(), 
  transforms.Resize((128, 128)),
  transforms.ToTensor(),] )


dataset = datasets.ImageFolder(file_path, transform=transforms)
print(dataset.classes)
print(dataset.class_to_idx)
batch_size=64
train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


class MyClassifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.batch= nn.BatchNorm2d(3)
    self.conv0 = nn.Conv2d(3, 64, 3) # 126
    self.maxp0 = nn.MaxPool2d(3)  # 41
    self.relu0 = nn.ReLU()
    self.dropout0 = nn.Dropout2d(0.5)  
    self.conv1 = nn.Conv2d(64, 32, 3) # 39
    self.maxp1 = nn.MaxPool2d(3) # 13
    self.relu1 = nn.ReLU() # 
    self.dropout1 = nn.Dropout2d(0.3)
    self.conv2 = nn.Conv2d(32, 16, 2) # 12
    self.maxp2 = nn.MaxPool2d(2) # 6
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout2d(0.2)
    self.flatten = nn.Flatten() # 6 * 6
    self.linear = nn.Linear(16 * 6 * 6, 1) 
  def forward(self, x):
    x = self.batch(x)
    x = self.conv0(x)
    x = self.maxp0(x)
    x = self.relu0(x)
    x = self.conv1(x)
    x = self.maxp1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.conv2(x)
    x = self.maxp2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    x = self.flatten(x)
    logits = self.linear(x)
    return logits


epochs = 10
loss_fn = nn.BCEWithLogitsLoss()
model = MyClassifier().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
metric = BinaryAccuracy().to(device)
train_acc ,train_loss = 0.0, 0.0
for e in range(epochs):
  model.train()
  for X, y in train_dataloader:
    X = X.to(device)
    pred = model(X)
    #print(pred)
    #break
    y = y.unsqueeze(1).float().to(device)
    loss = loss_fn(pred, y)
    #print(pred.shape)
    #break
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
   
    acc = metric(pred, y)
    
    train_acc += acc
    train_loss += loss

  print(f"{e}th batch acc: {train_acc / (len(train_dataloader) * (e + 1))}") 
  print(f"{e}th batch loss: {train_loss / (len(train_dataloader) * (e + 1))}") 

model.eval()
test_acc_avg, test_loss_avg = 0.0, 0.0
with torch.no_grad():
  for X, y in test_dataloader:
    X, y = X.to(device), y.to(device)
    pred = model(X)
    loss = loss_fn(pred, y.unsqueeze(1).float())
    test_acc_avg += metric(pred, y.unsqueeze(1).float())
    test_loss_avg += loss
print(f"test avg acc: {test_acc_avg / len(test_dataloader)}") 
print(f"test avg loss: {test_loss_avg / len(test_dataloader)}") 
    


  







