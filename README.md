# DA6401 Assignment 2 — Visual Perception Pipeline

## WandB Report
[Public WandB Report](https://api.wandb.ai/links/ae22b006-iitm-ac-in/4t4pfoaj

## GitHub Repo
[GitHub](https://github.com/Karthik-dot18/da6401_assignment_2)

## Project Structure
```
checkpoints/        ← model weights (stored on Drive, not committed)
data/
  pets_dataset.py   ← dataset classes
losses/
  iou_loss.py       ← IoULoss with mean/sum reduction
models/
  layers.py         ← CustomDropout
  vgg11.py          ← VGG11
  classification.py
  localization.py
  segmentation.py
multitask.py        ← MultiTaskPerceptionModel
train.py
inference.py
requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
# Task 1 - Classification
python train.py --task 1 --data_root /path/to/oxford-iiit-pet

# Task 2 - Localisation
python train.py --task 2 --data_root /path/to/oxford-iiit-pet --ckpt checkpoints/classifier.pth

# Task 3 - Segmentation
python train.py --task 3 --data_root /path/to/oxford-iiit-pet --ckpt checkpoints/classifier.pth --freeze full_finetune
```

## Inference
```bash
python inference.py --img /path/to/image.jpg
```
