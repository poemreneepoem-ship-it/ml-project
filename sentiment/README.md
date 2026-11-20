# Korean Sentence Emotion Classifier (PyTorch)

`learn.py` trains a model that reads a Korean sentence and predicts the emotion it expresses. It is a multiclass classifier over six labels.

## Data

File: `aihub_copus.csv`.

- `문장`: the sentence (input feature)
- `감정`: the emotion label (target)

Labels and approximate counts: 불안 (anxiety) 9,320, 분노 (anger) 9,160, 상처 (hurt) 9,143, 슬픔 (sadness) 9,125, 당황 (embarrassment) 8,756, 기쁨 (joy) 6,126. The file has 51,630 rows, and about 51,575 remain after cleaning.

## What the script does

1. Cleaning. Removes characters other than Hangul (including Jamo), digits, and punctuation. Collapses repeated spaces, drops empty sentences, and removes duplicate sentences.
2. Split. Divides the data into 80% train, 10% validation, and 10% test. The split is stratified, so each label keeps the same proportion in every part.
3. Tokenization. Splits each sentence into morphemes with Mecab (Korean morphological analyzer) and removes a fixed list of particles and short function words (`STOPWORDS`).
4. Vocabulary. Builds the word list from the training split only, keeping the most frequent `VOCAB_SIZE` words. Index 0 is unknown words and index 1 is padding.
5. Sequences. Converts each sentence to a list of indices, padded or truncated to `MAX_LEN` (the 99th percentile of training lengths).
6. Model. An embedding layer, then a 2-layer bidirectional LSTM. Mean and max pooling over the time steps are concatenated and passed through a linear layer to produce six scores.
7. Training. Cross-entropy loss with class weights (so rarer labels count more). Adam optimizer, gradient clipping, learning-rate reduction when validation loss stalls, and early stopping on validation loss.
8. Evaluation. Loads the best checkpoint and reports test loss, accuracy, a per-class precision/recall/F1 report, and a confusion matrix.

## Requirements

- Python 3.8 or later (the original project used the environment `pytorch-38`)
- torch
- pandas, numpy, scikit-learn
- konlpy, with the Mecab backend installed (`mecab.sh` in this folder is the setup script used earlier; Mecab must be installed before konlpy can use it)

## How to run

Run from the folder that contains `./`, because the paths are relative:

```
python learn.py
```


On Windows PowerShell, set the variable first with `$env:EPOCHS=2`, then run the script.

The script uses the GPU if one is available, otherwise the CPU.

## Outputs

- `sentiment.pth`: the best model, saved with its vocabulary, class names, and max length. The file is overwritten each time a better validation loss is found during training.
- Printed output: the training log for each epoch, and the final test report.


## Fine-tuning script (finetune_klue.py)

`finetune_klue.py` fine-tunes a Korean pre-trained model, `klue/roberta-base`, on the same six-emotion task. Every weight in the pre-trained model is updated, so this is the most accurate of the three approaches but needs the most memory and time.

What differs from `learn.py`:

- No Mecab. The pre-trained tokenizer splits the text, so only whitespace is normalized and duplicates are removed. Punctuation and Jamo are kept.
- The model is a pre-trained encoder with a new 6-output classification head.
- Training uses AdamW with a learning rate of 2e-5, a linear warmup over the first 10% of steps, and gradient clipping.
- Early stopping is on validation loss, with patience 2. The default is 3 epochs.
- The best model is saved to `klue_emotion_finetuned/` (about 440 MB).

Run it the same way as the other scripts:

```
python finetune_klue.py
EPOCHS=1 python finetune_klue.py   # quick check
```

Settings are at the top of the file: `PRETRAINED`, `OUT_DIR`, `BATCH`, `LR`, `EPOCHS`, `WARMUP_RATIO`, and `PATIENCE`.

## LoRA script (lora_klue.py)

`lora_klue.py` uses the same pre-trained model, but freezes it. It adds small trainable LoRA adapters (LoRA, low-rank adaptation) to the attention and feed-forward layers and trains only those adapters plus the classification head. About 1% of the parameters are trained.

Advantages compared with full fine-tuning: less memory and faster training, and the saved output is small (only the adapter, not the whole model). The trade-off is that accuracy is sometimes slightly lower than full fine-tuning.

Settings specific to LoRA:

| Setting | Default | Meaning |
|---|---|---|
| `LORA_R` | 16 | Rank of the adapter matrices (higher = more capacity, more parameters) |
| `LORA_ALPHA` | 32 | Scaling factor for the adapter output |
| `LORA_DROPOUT` | 0.1 | Dropout applied inside the adapters |
| `LORA_TARGETS` | query, key, value, dense | Layers that receive adapters |
| `LR` | 2e-4 | Higher than full fine-tuning, as usual for LoRA |
| `EPOCHS` | 5 | Maximum epochs |

Run it the same way:

```
python lora_klue.py
EPOCHS=1 python lora_klue.py   # quick check
```

The output folder `klue_emotion_lora/` contains the adapter (about 13 MB), the tokenizer, and `label_classes.json`. To use the model later, load the base model `klue/roberta-base` and attach the adapter with `PeftModel.from_pretrained(base, "klue_emotion_lora")`. The base model is downloaded again, so keep it available.

Requirements for these two scripts: `torch`, `transformers`, `scikit-learn`, `pandas`, and additionally `peft` for the LoRA script. These scripts do not need konlpy or Mecab.

## Comparing the three approaches

| Script | Model | Tokenizer | What is trained | Output size |
|---|---|---|---|---|
| `learn.py` | BiLSTM, trained from scratch | Mecab | All weights | Small |
| `finetune_klue.py` | klue/roberta-base | Pre-trained | All weights | About 440 MB |
| `lora_klue.py` | klue/roberta-base + LoRA | Pre-trained | Adapters and head only | About 13 MB (adapter only) |

Each script prints the test accuracy and confusion matrix at the end, so the results can be compared directly. All three use the same split, so the numbers are comparable.

## Notes and limitations

- Korean tokenization depends on Mecab. If Mecab is not available, the script will fail at the tokenization step.
- The test set is used only once, at the end. Repeated tuning should use the validation split.
