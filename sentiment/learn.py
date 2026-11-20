import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from konlpy.tag import Mecab
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = 'aihub_copus.csv'
STMT = '문장'
SNTM = '감정'
STOPWORDS = {'의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과',
             '도', '를', '으로', '자', '에', '와', '한', '하다'}
PAD, UNK = 1, 0                 # index 0 = unknown word, index 1 = padding
PAD_TOKEN, UNK_TOKEN = '<pad>', '<unk>'
VOCAB_SIZE = 10000
EMBED_DIM = 128
HIDDEN = 64
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH = 128
LR = 1e-3
EPOCHS = int(os.environ.get('EPOCHS', 50))   
PATIENCE = 4
SEED = 43
MODEL_PATH = 'sentiment.pth'              

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print('device:', device)

df = pd.read_csv(DATA_PATH)
print('raw rows:', len(df))
df = df.dropna(subset=[STMT, SNTM])
df[STMT] = df[STMT].astype(str).str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣 ]', '', regex=True)
df[STMT] = df[STMT].str.replace(r'\s+', ' ', regex=True).str.strip()
df = df[df[STMT] != ''].drop_duplicates(subset=[STMT]).reset_index(drop=True)
print('cleaned rows:', len(df))
print(df[SNTM].value_counts().to_string())

le = LabelEncoder()
y_all = le.fit_transform(df[SNTM])
classes = [str(c) for c in le.classes_]
sentences = df[STMT].tolist()

idx = np.arange(len(df))
tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=SEED, stratify=y_all)
tr_idx, va_idx = train_test_split(tr_idx, test_size=1/9, random_state=SEED, stratify=y_all[tr_idx])
print('train/val/test:', len(tr_idx), len(va_idx), len(te_idx))

tagger = Mecab()

def tokenize(sentence):
    return [t for t in tagger.morphs(sentence) if t not in STOPWORDS]

tok_tr = [tokenize(sentences[i]) for i in tr_idx]
tok_va = [tokenize(sentences[i]) for i in va_idx]
tok_te = [tokenize(sentences[i]) for i in te_idx]

counts = Counter(t for toks in tok_tr for t in toks)
word2idx = {UNK_TOKEN: UNK, PAD_TOKEN: PAD}
for w, _ in counts.most_common(VOCAB_SIZE):
    word2idx[w] = len(word2idx)
NUM_EMB = len(word2idx)
print('embedding rows:', NUM_EMB)

lengths = [len(t) for t in tok_tr]
MAX_LEN = max(1, int(np.percentile(lengths, 99)))
print('max_len (99th pct):', MAX_LEN)

def encode(token_lists):
    out = np.full((len(token_lists), MAX_LEN), PAD, dtype=np.int64)
    for i, toks in enumerate(token_lists):
        ids = [word2idx.get(t, UNK) for t in toks] or [UNK]   # never an empty sequence
        ids = ids[:MAX_LEN]
        out[i, :len(ids)] = ids
    return torch.from_numpy(out)

X_tr, X_va, X_te = encode(tok_tr), encode(tok_va), encode(tok_te)
y_tr, y_va, y_te = y_all[tr_idx], y_all[va_idx], y_all[te_idx]

def make_loader(X, y, shuffle):
    ds = TensorDataset(X, torch.from_numpy(np.asarray(y, dtype=np.int64)))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)

train_dl = make_loader(X_tr, y_tr, shuffle=True)
val_dl = make_loader(X_va, y_va, shuffle=False)
test_dl = make_loader(X_te, y_te, shuffle=False)

bincount = np.bincount(y_tr, minlength=len(classes))
class_w = torch.tensor(len(y_tr) / (len(classes) * bincount), dtype=torch.float32, device=device)
print('class weights:', [round(float(w), 3) for w in class_w])

class SentimentLSTM(nn.Module):
    def __init__(self, num_emb, emb_dim, hidden, num_layers, dropout, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_emb, emb_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2 * 2, num_classes)   # mean-pool + max-pool, both directions

    def forward(self, x):
        mask = (x != PAD)                                  # B x L, True for real tokens
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)                            # B x L x 2H
        m = mask.unsqueeze(-1).float()
        mean = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        mx = out.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1)[0]
        h = torch.cat([mean, mx], dim=1)
        return self.fc(self.dropout(h))                    # logits

model = SentimentLSTM(NUM_EMB, EMBED_DIM, HIDDEN, NUM_LAYERS, DROPOUT, len(classes)).to(device)
criterion = nn.CrossEntropyLoss(weight=class_w)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

def run_epoch(loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * X.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += X.size(0)
    return total_loss / n, correct / n

best_val, bad_epochs = float('inf'), 0
for epoch in range(EPOCHS):
    tr_loss, tr_acc = run_epoch(train_dl, optimizer)
    va_loss, va_acc = run_epoch(val_dl)
    scheduler.step(va_loss)
    print(f'epoch {epoch + 1}: train loss={tr_loss:.4f} acc={tr_acc:.4f} | '
          f'val loss={va_loss:.4f} acc={va_acc:.4f} | lr={optimizer.param_groups[0]["lr"]:.1e}')
    if va_loss < best_val:
        best_val, bad_epochs = va_loss, 0
        torch.save({'state_dict': model.state_dict(), 'word2idx': word2idx,
                    'classes': classes, 'max_len': MAX_LEN}, MODEL_PATH)
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print('early stopping')
            break

ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(ckpt['state_dict'])
test_loss, test_acc = run_epoch(test_dl)
model.eval()
preds = []
with torch.no_grad():
    for X, _ in test_dl:
        preds.append(model(X.to(device)).argmax(dim=1).cpu())
y_pred = torch.cat(preds).numpy()

print(f'\nTEST loss={test_loss:.4f} accuracy={accuracy_score(y_te, y_pred):.4f}')
print(classification_report(y_te, y_pred, target_names=classes, digits=4))
cm = pd.DataFrame(confusion_matrix(y_te, y_pred), index=classes, columns=classes)
print('Confusion matrix (rows = true, cols = predicted):')
print(cm.to_string())

with open('result/label_classes.json', 'w', encoding='utf-8') as f:
    json.dump(classes, f, ensure_ascii=False)
