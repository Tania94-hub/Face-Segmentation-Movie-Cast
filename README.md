# Real-Time Face Segmentation for Movie Cast Identification

## Project Overview
Deep learning pipeline to detect and segment faces in movie scene screenshots 
using U-Net architecture. Built as GUVI HCL Final Project.

## Problem Statement
Company X's streaming app needs to automatically detect and segment faces 
in movie scene screenshots so users can pause videos and instantly view 
cast/crew details for actors on screen.

## Tech Stack
- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Albumentations
- Scikit-learn

## Dataset
- 409 movie scene images with face bounding box annotations
- Images stored as NumPy object arrays with embedded annotation metadata
- Each entry contains image array + normalized face coordinates
- Source: GUVI HCL provided dataset

## Approach

### 1. Data Preprocessing & EDA
- Loaded images from .npy object array format
- Extracted face bounding box annotations from embedded metadata
- Generated binary face masks from normalized coordinates
- Resized all images and masks to 256×256
- Visualized sample images, masks and face region distribution

### 2. Model Building
Three U-Net architectures were trained and compared:
- **MobileNetV2 U-Net** — pretrained encoder, frozen weights
- **Custom U-Net** — built from scratch with dropout regularization
- **VGG16 U-Net** — pretrained VGG16 encoder with fine-tuning

### 3. Training Strategy
- Custom Dice Loss + Binary Crossentropy combined loss
- Custom Dice Coefficient and IoU metrics
- ModelCheckpoint, EarlyStopping, ReduceLROnPlateau callbacks
- Albumentations for data augmentation (3x dataset size)
- Train/Val split: 80/20

### 4. Evaluation
Evaluated on validation set across all models.

## Model Results

| Model | Val Dice | Val IoU | Augmentation | Best Epoch |
|-------|----------|---------|--------------|------------|
| MobileNetV2 U-Net | 0.679 | 0.517 | None | 6 |
| Custom U-Net | 0.594 | 0.426 | Basic flip/brightness | 90 |
| VGG16 U-Net | 0.672 | 0.466 | Albumentations 3x | 10 |
| VGG16 Fine-tuned | 0.652 | 0.470 | Albumentations 3x | 7 |

## Evaluation Metrics

| Metric | Score | Target |
|--------|-------|--------|
| Dice Coefficient | 0.6173 | >0.92 |
| IoU | 0.4701 | >0.88 |
| F1 Score | 0.6173 | >0.90 |
| Avg Inference Speed | 237.3 ms | <100ms |

## Key Observations
1. Dataset size (409 images) was the primary limiting factor — train dice reached 0.91 confirming the model learns the task, but validation dice capped at ~0.68 due to overfitting
2. MobileNetV2 U-Net performed best (val_dice 0.679) despite no augmentation
3. Albumentations augmentation tripled training data but did not significantly improve validation dice — suggesting more diverse original data is needed
4. Fine-tuning VGG16 encoder with LR=1e-5 did not improve over frozen encoder
5. Custom U-Net from scratch underperformed pretrained encoders on this small dataset
6. Inference speed of 237ms is above 100ms target for single-image CPU prediction

## Project Structure
├── notebook.ipynb        # EDA, preprocessing, model training, evaluation
├── requirements.txt      # Dependencies
└── README.md
