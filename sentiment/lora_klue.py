import json
import os

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

DATA_PATH = 'aihub_copus.csv'
STMT = '문장'
SNTM = '감정'
PRETRAINED = 'klue/roberta-base'           
OUT_DIR = 'klue_emotion_lora'              
MAX_LEN_CAP = 128
BATCH = 32
LR = 2e-4                                  
WEIGHT_DECAY = 0.01
EPOCHS = int(os.environ.get('EPOCHS', 5))  
WARMUP_RATIO = 0.1
PATIENCE = 2
SEED = 43
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGETS = ['query', 'key', 'value', 'dense']   

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print('device:', device)

df = pd.read_csv(DATA_PATH)
print('raw rows:', len(df))
df = df.dropna(subset=[STMT, SNTM])
df[STMT] = df[STMT].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
df = df[df[STMT] != ''].drop_duplicates(subset=[STMT]).reset_index(drop=True)
print('cleaned rows:', len(df))

le = LabelEncoder()
y_all = le.fit_transform(df[SNTM])
classes = [str(c) for c in le.classes_]
sentences = df[STMT].tolist()

idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=SEED, stratify=y_all)
tr_idx, va_idx = train_test_split(tr_idx, test_size=1/9, random_state=SEED, stratify=y_all[tr_idx])
print('train/val/test:', len(tr_idx), len(va_idx), len(te_idx))

tokenizer = AutoTokenizer.from_pretrained(PRETRAINED)
lengths = [len(x) for x in tokenizer([sentences[i] for i in tr_idx])['input_ids']]
MAX_LEN = min(MAX_LEN_CAP, int(np.percentile(lengths, 99)))
print('max_len:', MAX_LEN)

def make_loader(ids, shuffle):
    enc = tokenizer([sentences[i] for i in ids], truncation=True, max_length=MAX_LEN,
                    padding='max_length', return_tensors='pt')
    labels = torch.tensor(y_all[ids], dtype=torch.long)
    data = list(zip(enc['input_ids'], enc['attention_mask'], labels))
    return DataLoader(data, batch_size=BATCH, shuffle=shuffle)

train_dl = make_loader(tr_idx, shuffle=True)
val_dl = make_loader(va_idx, shuffle=False)
test_dl = make_loader(te_idx, shuffle=False)

counts = np.bincount(y_all[tr_idx], minlength=len(classes))
class_w = torch.tensor(len(tr_idx) / (len(classes) * counts), dtype=torch.float32, device=device)
criterion = nn.CrossEntropyLoss(weight=class_w)

base = AutoModelForSequenceClassification.from_pretrained(
    PRETRAINED, num_labels=len(classes),
    id2label={i: c for i, c in enumerate(classes)},
    label2id={c: i for i, c in enumerate(classes)})
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS,            # also makes the classification head trainable
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
    target_modules=LORA_TARGETS, bias='none')
model = get_peft_model(base, lora_cfg)
model.to(device)
model.print_trainable_parameters()

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=WEIGHT_DECAY)
total_steps = len(train_dl) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=int(WARMUP_RATIO * total_steps), num_training_steps=total_steps)

def run_epoch(loader, training):
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for input_ids, attn, y in loader:
            input_ids, attn, y = input_ids.to(device), attn.to(device), y.to(device)
            logits = model(input_ids=input_ids, attention_mask=attn).logits
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
        tokenizer.save_pretrained(OUT_DIR)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print('early stopping')
            break

base_eval = AutoModelForSequenceClassification.from_pretrained(PRETRAINED, num_labels=len(classes))
best_model = PeftModel.from_pretrained(base_eval, OUT_DIR).to(device)
best_model.eval()
preds, truths = [], []
with torch.no_grad():
    for input_ids, attn, y in test_dl:
        logits = best_model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
        preds.append(logits.argmax(dim=1).cpu())
        truths.append(y)
y_pred = torch.cat(preds).numpy()
y_true = torch.cat(truths).numpy()

print(f'\nTEST accuracy={accuracy_score(y_true, y_pred):.4f}')
print(classification_report(y_true, y_pred, target_names=classes, digits=4))
cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=classes, columns=classes)
print('Confusion matrix (rows = true, cols = predicted):')
print(cm.to_string())

with open(os.path.join(OUT_DIR, 'label_classes.json'), 'w', encoding='utf-8') as f:
    json.dump(classes, f, ensure_ascii=False)
