import json
import os
import glob

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torchvision.models as models
from torchvision import transforms, datasets
from torchvision.transforms import v2
from torch.utils.data import random_split, DataLoader
from torchmetrics.classification import BinaryAccuracy
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import ssl
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

ssl._create_default_https_context = ssl._create_unverified_context
# pretrained model with ImageNet
PRETRAINED_MODEL = 'microsoft/resnet-18'
LORA_R = 8
LORA_ALPHA = 8
LORA_TARGETS = ['convolution']
LORA_DROPOUT = 0.2
LR = 1e-3
EPOCHS = int(os.environ.get('EPOCHS', 5))  
WARMUP_RATIO = 0.1
PATIENCE = 2
BATCH=32
OUT_DIR = 'resnet_catdog_lora'              
IMAGE_PATH= "./dataset/versions/1/PetImages"
image_processor = AutoImageProcessor.from_pretrained(PRETRAINED_MODEL)

# validation of images

def is_valid_image(file_path: str) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()  
        with Image.open(file_path) as img:
            img.load()    
        return True
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return False

image_file_paths = glob.glob(IMAGE_PATH + "/**/*.*", recursive=True)
for f in image_file_paths:
  if not is_valid_image(f):
    print(f'Deleting file {f}...')
    os.remove(f)


all_dataset = datasets.ImageFolder(IMAGE_PATH)
classes = all_dataset.classes
class_to_idx = all_dataset.class_to_idx
idx_to_class = {i: c for i, c in enumerate(classes)}

generator = torch.Generator().manual_seed(43)
train_dataset, val_dataset, test_dataset = random_split(all_dataset, [0.8, 0.1, 0.1], generator=generator)
def make_loader(dataset, shuffle=False):
  items = []
  for img, label in dataset:
    pv = image_processor(img, return_tensors='pt')['pixel_values'][0]
    items.append((pv, label))
 
  return DataLoader(items, batch_size=BATCH, shuffle=shuffle)
  
val_dl = make_loader(val_dataset)
train_dl = make_loader(train_dataset, True)
test_dl = make_loader(test_dataset)
criterion = nn.CrossEntropyLoss()

base = AutoModelForImageClassification.from_pretrained(
    PRETRAINED_MODEL, num_labels=len(classes),
    id2label=idx_to_class, label2id=class_to_idx, ignore_mismatched_sizes=True)

lora_cfg = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION, # also makes the classification head trainable
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS, bias='none', 
    modules_to_save=['classifier'])

model = get_peft_model(base, lora_cfg)
model.print_healthcheck()
model = model.to(device)
model.print_trainable_parameters()
    

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable, lr=LR)
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps)

def run_epoch(loader, training):
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(pixel_values=X).logits
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += y.size(0)
    return total_loss / n, correct / n


os.makedirs(OUT_DIR, exist_ok=True)
best_val, bad_epochs = float('inf'), 0
for epoch in range(EPOCHS):
    tr_loss, tr_acc = run_epoch(train_dl, training=True)
    va_loss, va_acc = run_epoch(val_dl, training=False)
    print(f'epoch {epoch + 1}: train loss={tr_loss:.4f} acc={tr_acc:.4f} | '
          f'val loss={va_loss:.4f} acc={va_acc:.4f}')
    if va_loss < best_val:
        best_val, bad_epochs = va_loss, 0
        model.save_pretrained(OUT_DIR)       # LoRA adapter + classification head
        image_processor.save_pretrained(OUT_DIR)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print('early stopping')
            break

base_eval = AutoModelForImageClassification.from_pretrained(PRETRAINED_MODEL, num_labels=len(classes), ignore_mismatched_sizes=True)
best_model = PeftModel.from_pretrained(base_eval, OUT_DIR).to(device)
best_model.eval()
preds, truths = [], []
with torch.no_grad():
    for X, y in test_dl:
        X, y = X.to(device), y.to(device)
        logits = best_model(pixel_values=X).logits
        preds.append(logits.argmax(dim=1).cpu())
        truths.append(y)
y_pred = torch.cat(preds).cpu().numpy()
y_true = torch.cat(truths).cpu().numpy()

print(f'\nTEST accuracy={accuracy_score(y_true, y_pred):.4f}')
print(classification_report(y_true, y_pred, target_names=classes, digits=4))
cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=classes, columns=classes)
print('Confusion matrix (rows = true, cols = predicted):')
print(cm.to_string())

with open(os.path.join(OUT_DIR, 'label_classes.json'), 'w', encoding='utf-8') as f:
    json.dump(classes, f, ensure_ascii=False)
