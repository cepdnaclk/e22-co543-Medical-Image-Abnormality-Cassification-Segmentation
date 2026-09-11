# Medical Image Processing & Deep Learning Codebase

This directory houses the complete computational pipeline for **Brain Tumor MRI 4-Class Classification, Preprocessing Ablation, and Grad-CAM Visual Explainability**, developed for the **CO543 Image Processing** project at the **University of Peradeniya**.

---

## Directory Organization

```
code/
├── Updated notebook with comparison/            # Primary experimental notebook & ablation weights
│   ├── Brain_Tumor_MRI_Classification_and_Explainability (1).ipynb   # Comprehensive master notebook
│   ├── best_resnet18 (1).pth                    # ResNet-18 best checkpoint (Full pipeline: 95.69% Acc)
│   ├── best_efficientnet_b0 (1).pth             # EfficientNet-B0 best checkpoint (95.62% Acc, -64% Params)
│   ├── best_resnet18_pureraw.pth                # Ablation Stage 1: Pure raw baseline (73.19% Acc)
│   └── best_resnet18_raw_aug.pth                # Ablation Stage 2: Raw + Augmentation (93.31% Acc)
├── New model/                                   # Initial development & baseline exploration
│   ├── Brain_Tumor_MRI_Classification_and_Explainability.ipynb       # First-generation training notebook
│   ├── best_resnet18.pth                        # ResNet-18 checkpoint (Full pipeline)
│   ├── best_resnet18_raw.pth                    # ResNet-18 checkpoint (Raw training baseline)
│   └── best_efficientnet_b0.pth                 # EfficientNet-B0 checkpoint
├── preprocessing/                               # Computer vision engineering & exploratory analysis
│   └── preprocessing-pipeline-visulaization.ipynb  # Step-by-step visual inspection of OpenCV pipeline
└── README.md                                    # This codebase guide & execution manual
```

---

## Checkpoint Registry (`.pth` Models)

All trained model weights were converged using PyTorch on an **NVIDIA RTX 6000 Ada Generation (51,557 MiB VRAM)** workstation.

| Checkpoint File | Directory | Backbone | Experimental Condition | Size | Test Accuracy | Macro F1 | Description |
|---|---|---|---|---|---|---|---|
| `best_resnet18 (1).pth` | `Updated notebook with comparison/` | ResNet-18 | Full Preprocessing + Albumentations | 44.8 MB | **95.69%** | **0.9563** | Production model; highest overall diagnostic accuracy. |
| `best_efficientnet_b0 (1).pth` | `Updated notebook with comparison/` | EfficientNet-B0 | Full Preprocessing + Albumentations | 16.4 MB | **95.62%** | **0.9555** | Ultra-efficient edge model (-64.1% params, -78.5% FLOPs). |
| `best_resnet18_raw_aug.pth` | `Updated notebook with comparison/` | ResNet-18 | Raw Scans + Albumentations | 44.8 MB | **93.31%** | **0.9324** | Ablation Stage 2: Augmentation without OpenCV preprocessing. |
| `best_resnet18_pureraw.pth` | `Updated notebook with comparison/` | ResNet-18 | Pure Raw Scans (No Preproc / No Aug) | 44.8 MB | **73.19%** | **0.7301** | Ablation Stage 1: Minimalist baseline without domain adaptations. |
| `best_resnet18.pth` | `New model/` | ResNet-18 | Full Preprocessing + Albumentations | 44.8 MB | 95.69% | 0.9563 | Initial generation full-pipeline checkpoint. |
| `best_resnet18_raw.pth` | `New model/` | ResNet-18 | Raw Baseline | 44.8 MB | 73.19% | 0.7301 | Initial generation unaugmented baseline. |
| `best_efficientnet_b0.pth` | `New model/` | EfficientNet-B0 | Full Preprocessing + Albumentations | 16.4 MB | 95.62% | 0.9555 | Initial generation EfficientNet-B0 weights. |

---

## Preprocessing Pipeline Architecture

The core computer vision engineering resides in both `preprocessing/preprocessing-pipeline-visulaization.ipynb` and `Updated notebook with comparison/Brain_Tumor_MRI_Classification_and_Explainability (1).ipynb`. The pipeline mitigates non-biological MRI artifacts (scanner variations, skull margins, low tissue contrast) before training.

```
[Raw MRI Scan] ──> [1. Contour Crop] ──> [2. Bilateral Filter] ──> [3. LAB CLAHE] ──> [4. Resize (224x224)] ──> [5. Normalize]
```

### 1. Contour-Based Skull Stripping (`crop_brain_contour`)
Isolates brain tissue and discards peripheral black background pixels and skull borders:
```python
def crop_brain_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Otsu thresholding + binary threshold to isolate tissue mask
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Extract largest connected component contour
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if not cnts:
        return image
    c = max(cnts, key=cv2.contourArea)
    
    # Crop bounding rectangle
    x, y, w, h = cv2.boundingRect(c)
    return image[y:y+h, x:x+w]
```

### 2. Edge-Preserving Denoising
Gaussian noise and acquisition grain are attenuated without blurring critical tumor-tissue boundaries:
```python
denoised = cv2.bilateralFilter(cropped, d=5, sigmaColor=50, sigmaSpace=50)
```

### 3. Perceptual Contrast Enhancement (LAB CLAHE)
Standard RGB equalization distorts color-neutral MRI tissue. We project to the **CIE LAB** color space, apply Contrast Limited Adaptive Histogram Equalization to the luminance channel ($L^*$), and project back to RGB:
```python
lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
cl = clahe.apply(l)
enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
```

### 4. Training Augmentation Suite (`Albumentations`)
To enforce rotational, scale, and sensor invariance without generating unrealistic medical deformations:
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])
```

---

## Preprocessing Ablation Study

To mathematically demonstrate the contribution of each component, three controlled training runs were performed under identical hyperparameters (ResNet-18, 15 epochs, seed 42):

```
┌─────────────────────────────────────────────────────────────┐
│ ABLATION ACCURACY GAIN                                      │
├─────────────────────────────────────────────────────────────┤
│ Stage 1: Pure Raw Baseline              73.19%              │
│ Stage 2: Raw Scans + Albumentations     93.31% (+20.12%)    │
│ Stage 3: Full Preprocessing Pipeline    95.69% (+22.50%)    │
└─────────────────────────────────────────────────────────────┘
```

- **Stage 1 (`best_resnet18_pureraw.pth`)**: Raw scans passed directly to the network without skull stripping, denoising, or augmentation. Severe overfitting occurred, peaking at **73.19%** test accuracy.
- **Stage 2 (`best_resnet18_raw_aug.pth`)**: Albumentations applied directly to uncropped, noisy scans. Accuracy jumped by **+20.12%** to **93.31%**, proving that geometric and photometric augmentations significantly reduce regularization error.
- **Stage 3 (`best_resnet18 (1).pth`)**: Integration of contour cropping, bilateral filtering, and LAB CLAHE with Albumentations. Delivered an additional **+2.38%** boost to **95.69%**, confirming that edge-preserving enhancement and background elimination provide essential diagnostic signal.

---

## Model Benchmark: ResNet-18 vs EfficientNet-B0

Both architectures were trained with ImageNet pretrained backbones, fine-tuned on the 4-class brain tumor classification task:

| Evaluation Metric | ResNet-18 (`best_resnet18 (1).pth`) | EfficientNet-B0 (`best_efficientnet_b0 (1).pth`) | Delta / Tradeoff |
|---|---|---|---|
| **Test Accuracy** | **95.69%** (1531 / 1600) | 95.62% (1530 / 1600) | -0.07% |
| **Macro F1 Score** | **0.9563** | 0.9555 | -0.0008 |
| **Glioma F1** | **0.923** | 0.921 | -0.002 |
| **Meningioma F1** | **0.932** | 0.931 | -0.001 |
| **Pituitary F1** | **0.985** | **0.985** | Parity |
| **No Tumor F1** | **0.985** | 0.984 | -0.001 |
| **Parameters** | 11,180,612 (11.18M) | **4,012,672 (4.01M)** | **-64.1% reduction** |
| **FLOPs (MACs)** | 1.81 GFLOPs | **0.39 GFLOPs** | **-78.5% reduction** |
| **Model Size** | 44.8 MB | **16.4 MB** | **-63.4% smaller** |
| **Inference Latency** | 5.04 ms / image | 5.40 ms / image | +0.36 ms (Depthwise overhead) |

### Key Engineering Takeaway
**EfficientNet-B0 achieves near-identical diagnostic accuracy (95.62% vs 95.69%) with a 64.1% parameter reduction and a 78.5% compute reduction**, making it the optimal candidate for embedded edge hardware and mobile radiology stations.

---

## Grad-CAM Visual Explainability

To satisfy clinical auditability, Gradient-weighted Class Activation Mapping (Grad-CAM) is integrated into the inference pipeline:

1. **Target Feature Maps ($A^k$)**:
   - ResNet-18: Final convolution block `model.layer4[-1]` ($7 \times 7 \times 512$).
   - EfficientNet-B0: Top feature block `model.features[-1]` ($7 \times 7 \times 1280$).
2. **Gradient Importance ($\alpha_k^c$)**:
   $$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial y^c}{\partial A_{i, j}^k}$$
3. **Heatmap Generation ($L_{\text{Grad-CAM}}^c$)**:
   $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
4. **Bilinear Overlay**: The heatmap is normalized, upsampled to $224 \times 224$, colorized with `cv2.COLORMAP_JET`, and superimposed onto the original MRI with $\alpha = 0.5$.

---

## Quickstart: How to Run Inference

### 1. Environment Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python albumentations timm torchinfo scikit-learn matplotlib seaborn imutils
```

### 2. Loading Weights & Predicting on Scans
```python
import torch
import cv2
import numpy as np
from torchvision import models
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. Define device and class labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

# 2. Reconstruct ResNet-18 architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

# 3. Load checkpoint
checkpoint_path = "Updated notebook with comparison/best_resnet18 (1).pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# 4. Preprocess input scan
def preprocess_scan(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Bilateral filter & CLAHE
    denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
    lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2RGB)
    
    # Transform
    val_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    tensor = val_tf(image=enhanced)['image'].unsqueeze(0)
    return tensor.to(device)

# 5. Run inference
tensor = preprocess_scan("sample_mri.jpg")
with torch.no_grad():
    logits = model(tensor)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = np.argmax(probs)
    print(f"Prediction: {classes[pred_idx]} ({probs[pred_idx]*100:.2f}% confidence)")
```

---

## Hardware & Environment Specifications

- **OS / Platform**: Linux / Windows x64
- **Accelerator**: NVIDIA RTX 6000 Ada Generation (51,557 MiB VRAM)
- **CUDA Version**: 12.8 (Driver 572.70)
- **PyTorch**: 2.6.0+cu124
- **Mixed Precision**: FP16 (`torch.cuda.amp.autocast`)
- **Random Seed**: 42 (Enforced across NumPy, PyTorch, and CUDA)
