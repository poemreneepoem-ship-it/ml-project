# ML Project

This repository contains two PyTorch experiments. Each one is in its own subdirectory and can be run from inside that folder.

| Folder | Task | Scripts |
|---|---|---|
| `cat-dog-classification/` | Image: classify a photo as cat or dog (binary) | `learn.py`, `resnet.py` |
| `sentiment/` | Text: classify the emotion of a Korean sentence (6 labels) | `learn.py`, `finetune_klue.py`, `lora_klue.py`, `mecab.sh` |

## cat-dog-classification/

A binary image classifier trained on the Kaggle "Dog and Cat Classification Dataset".

- `learn.py`: a small CNN trained from scratch. Images are resized to 128x128. It has three convolution blocks with max pooling, BatchNorm and dropout, and one output unit trained with BCE loss. It trains for 10 epochs, then prints test loss and accuracy.
- `resnet.py`: transfer learning with ResNet-18 pretrained on ImageNet. Only `layer4` and a new final layer are trained; the rest is frozen. Images are 224x224 with ImageNet normalization. It trains for 20 epochs with a learning-rate scheduler, then prints test loss and accuracy.

Setup:

1. Download the dataset from Kaggle: `bhavikjikadara/dog-and-cat-classification-dataset`.
2. Make sure `./dataset/versions/1/PetImages/` exists and contains the `Cat` and `Dog` folders.

In this repository, `dataset` is a symlink to a path in one author's local kagglehub cache. That path does not exist on other machines, so replace the symlink with your own dataset location.

Run:

```
cd cat-dog-classification
python learn.py
python resnet.py
```

Note: `random_split` is not seeded in either script, so the train/test split changes on every run. `resnet.py` also disables SSL certificate checks globally to download the ImageNet weights.

## sentiment/

A classifier for six Korean emotion labels: 불안 (anxiety), 분노 (anger), 상처 (hurt), 슬픔 (sadness), 당황 (embarrassment), and 기쁨 (joy). The data is `aihub_copus.csv`, with 51,630 rows. All three training scripts use the same cleaning, the same 80/10/10 stratified split, and the same seed, so their test results can be compared.

- `learn.py`: a 2-layer bidirectional LSTM trained from scratch, with Mecab for Korean tokenization. The smallest model.
- `finetune_klue.py`: full fine-tuning of `klue/roberta-base`. Every weight is updated. The saved model is about 440 MB.
- `lora_klue.py`: LoRA adapters on `klue/roberta-base`. Only about 1% of the parameters are trained. The saved adapter is about 13 MB.
- `mecab.sh`: setup script that installs the Mecab Korean analyzer. `learn.py` needs it through konlpy. It supports Linux and macOS.

The full description of each script, its settings and its outputs is in `sentiment/README.md`.

Requirements: `torch`, `pandas`, `numpy`, `scikit-learn`. `learn.py` also needs `konlpy` with Mecab. The two RoBERTa scripts need `transformers`, and `lora_klue.py` also needs `peft`.

Run:

```
cd sentiment
python learn.py
python finetune_klue.py
python lora_klue.py
```

Note: `learn.py` writes `result/label_classes.json`, so create a `result/` folder before running it.

## License

See `LICENSE`.
