import torch
from torch import nn, optim
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader
from torchmetrics.classification import BinaryAccuracy
import ssl

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

ssl._create_default_https_context = ssl._create_unverified_context
# pretrained model with ImageNet
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
  param.requires_grad = False
num_features = model.fc.in_features 
model.fc = nn.Linear(num_features, 1)
for param in model.layer4.parameters():
  param.requires_grad = True
#model.fc.weight = nn.init.normal_(model.fc.weight, mean=0.0, std=0.01)    
#model.fc.bias = nn.init.zeros_(model.fc.bias)
#print(model)
#exit(0)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam([
{"params": model.fc.parameters(), "lr":1e-3},
{"params": model.layer4.parameters(), "lr":1e-5}])
num_epochs = 20
batch_size = 128

file_path = "./dataset/versions/1/PetImages"
transforms = transforms.Compose([
  v2.RGB(), 
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])] )
dataset = datasets.ImageFolder(file_path, transform=transforms)
train_dataset, test_dataset = random_split(dataset, [0.8, 0.2])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

metric = BinaryAccuracy()
metric = metric.to(device)

model.train()
model.bn1.eval()
model.layer1.eval()
model.layer2.eval()
model.layer3.eval()

# training
for e in range(num_epochs):
  train_avg_acc, train_avg_loss = 0.0, 0.0
  for X, y in train_dataloader:
    X = X.to(device)
    y = y.unsqueeze(1).float()
    y = y.to(device)
    pred = model(X)

    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    acc = metric(pred, y)
    train_avg_acc += acc
    train_avg_loss += loss.item()
  train_avg_loss /= len(train_dataloader)
  train_avg_acc /= len(train_dataloader)
  scheduler.step(train_avg_loss)

  print(f"{e+1}th loss = {train_avg_loss}")
  print(f"{e+1}th acc = {train_avg_acc}")
   

# test

model.eval()
with torch.no_grad():
  test_avg_acc, test_avg_loss = 0.0, 0.0
  for X, y in test_dataloader:
    X = X.to(device)
    y = y.unsqueeze(1).float()
    y = y.to(device)
    pred = model(X)
    loss = criterion(pred, y)
    test_avg_acc += metric(pred, y)
    test_avg_loss += loss.item()
  
  test_avg_acc /= len(test_dataloader)  
  test_avg_loss /= len(test_dataloader)
  
  print(test_avg_loss, test_avg_acc)
