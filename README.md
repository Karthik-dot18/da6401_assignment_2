# DA6401 Assignment 2 — Complete Visual Perception Pipeline

This repository contains the implementation for DA6401 – Introduction to Deep Learning (Assignment 2). The objective of this assignment was to build a complete multi-stage visual perception system using the Oxford-IIIT Pet Dataset.

The final pipeline performs three core computer vision tasks:

- Breed Classification (37 pet classes)
- Object Localization (bounding box prediction)
- Semantic Segmentation (pixel-wise trimap masks)

A unified multi-task model was developed using a shared VGG11 encoder, with separate task-specific heads for classification, localization, and segmentation.
## Implemented Components

### 1. VGG11 Classification Network

- Implemented **VGG11 from scratch** using PyTorch  
- Added **Batch Normalization** layers  
- Implemented custom **Dropout** without using `torch.nn.Dropout`  
- Trained for **37-class pet breed classification**

### 2. Object Localization

- Reused trained VGG11 backbone as encoder  
- Added regression head to predict bounding boxes:

  - `x_center`
  - `y_center`
  - `width`
  - `height`

- Implemented custom **IoU Loss**

### 3. U-Net Style Segmentation

- Used VGG11 encoder as contracting path  
- Built symmetric decoder using transpose convolutions  
- Used skip connections for feature fusion  
- Trained using **CrossEntropy + Dice Loss**

### 4. Unified Multi-Task Pipeline

Single forward pass returns:

- Classification logits  
- Bounding box coordinates  
- Segmentation mask
## WandB Report
[Public WandB Report](https://api.wandb.ai/links/ae22b006-iitm-ac-in/4t4pfoaj

## GitHub Repo
[GitHub](https://github.com/Karthik-dot18/da6401_assignment_2)



