---
layout: home
permalink: index.html
repository-name: e22-co543-Medical-Image-Abnormality-Cassification-Segmentation
title: Medical Image Abnormality Classification & Segmentation
---

# Medical Image Abnormality Classification and Segmentation System (MedImgSys)

[![Course](https://img.shields.io/badge/Course-CO5430%20Computer%20Vision-blue.svg)](http://www.ce.pdn.ac.lk/)
[![Group](https://img.shields.io/badge/Group-G14-green.svg)]()
[![Project ID](https://img.shields.io/badge/Project%20ID-P19-orange.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)

---

## Team Members

* **Hansara S. H. S.** (`E/22/130`) - [e22130@eng.pdn.ac.lk](mailto:e22130@eng.pdn.ac.lk)
* **H. M. Liyanage** (`E/22/211`) - [e22211@eng.pdn.ac.lk](mailto:e22211@eng.pdn.ac.lk)
* **D. M. N. N. Bandara** (`E/22/044`) - [e22044@eng.pdn.ac.lk](mailto:e22044@eng.pdn.ac.lk)
* **Weerasinghe W. P. T. H.** (`E/22/421`) - [e22421@eng.pdn.ac.lk](mailto:e22421@eng.pdn.ac.lk)

**Institution:** Department of Computer Engineering, Faculty of Engineering, University of Peradeniya  
**Target Organization:** Medical Image Analysis Research Group, Teaching Hospital Peradeniya

---

## Table of Contents

1. [Introduction & Clinical Background](#introduction--clinical-background)
2. [Dataset Specifications](#dataset-specifications)
3. [System Architecture & Conceptual Design](#system-architecture--conceptual-design)
4. [Methodology & Development Phases](#methodology--development-phases)
   - [Phase 1: Soft-Tissue Image Preprocessing](#phase-1-soft-tissue-image-preprocessing)
   - [Phase 2: Baseline Model (ResNet-18)](#phase-2-baseline-model-resnet-18)
   - [Phase 3: Comparative Architecture (EfficientNet-B0)](#phase-3-comparative-architecture-efficientnet-b0)
   - [Phase 4: Spatial Localization & Explainability (Grad-CAM)](#phase-4-spatial-localization--explainability-grad-cam)
5. [Benchmark Results & Experimental Evaluation](#benchmark-results--experimental-evaluation)
6. [Repository Structure](#repository-structure)
7. [Environment Setup & Installation](#environment-setup--installation)
8. [Conclusion & Key Achievements](#conclusion--key-achievements)
9. [Links](#links)

---

## Introduction & Clinical Background

The **Medical Image Analysis Research Group** at the **Teaching Hospital Peradeniya** handles critical diagnostic imaging cases (e.g., tumors, fractures, internal anomalies) and radiographic MRI/CT examinations. Currently, the department relies on manual visual inspection, physical radiological films, and human-based Diagnostic Imaging Reports (DIR).

This manual workflow creates critical operational bottlenecks:

* **Inefficient Anomaly Detection:** Manual visual inspection makes tracing subtle historical patient anomalies or matching microscopic pattern variations across physical films extremely slow.
* **Diagnostic Accuracy Vulnerabilities:** Tracking complex biological structures, tumor boundaries, lesions, and laboratory features across manual inspections risks misinterpretation or inter-observer diagnostic gaps.
* **Data Processing & Granular Image Segmentation:** Sensitive radiographic findings, patient anomaly details, and clinical scans lack strict automated pixel-level boundary restrictions.
* **Report Compilation Delay:** Assembling Automated Diagnostic Reports (ADR) for clinical submissions requires manual transcription from multiple unlinked imaging modalities.

**MedImgSys** is a deep learning system engineered specifically to automate, secure, and streamline the diagnostic imaging workflows of the Radiology Department. The primary objective of this project is to implement a robust convolutional neural network pipeline that guarantees high accuracy, enforces precise segmentation and spatial localization constraints, maintains performance logging via metric tracking, and optimizes model architectures for automated diagnostic report compilation.

---

## Dataset Specifications

* **Source:** Kaggle Brain Tumor MRI Dataset
* **Volume:** **7,022** high-resolution MRI Scans
* **Classes (4 Diagnostic Categories):**
  1. `Glioma`: Infiltrative glial cell tumors with irregular tissue boundaries.
  2. `Meningioma`: Tumors arising from meningeal membranes surrounding the brain.
  3. `Pituitary Tumor`: Neoplasms localized near the pituitary gland at the base of the brain.
  4. `No Tumor`: Healthy baseline MRI scans (normal controls).
* **Split Ratio:** Stratified **80% Training**, **10% Validation**, and **10% Testing** partition.

---

## System Architecture & Conceptual Design

The core of **MedImgSys** is designed around strict machine learning principles to ensure minimal false negatives, high diagnostic integrity, explicit feature extraction via specialized network architectures, and structured model optimization.

```
                                  [ Input MRI Scan (224x224) ]
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │ Phase 1: Soft-Tissue Image Preprocessing Pipeline   │
                     ├─────────────────────────────────────────────────────┤
                     │ 1. CLAHE (clipLimit=2.0, tileGridSize=8x8)          │
                     │ 2. Bilateral Filter (d=9, sigmaColor=75, space=75) │
                     │ 3. Albumentations (Affine, Flips, Contrast)        │
                     └──────────────────────────┬──────────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
     ┌───────────────────────────────────┐             ┌───────────────────────────────────┐
     │ Phase 2: ResNet-18 (Baseline CNN) │             │ Phase 3: EfficientNet-B0 (Modern) │
     ├───────────────────────────────────┤             ├───────────────────────────────────┤
     │ • Parameters: ~11.18M             │             │ • Parameters: ~4.01M (-64%)       │
     │ • Pre-trained ImageNet Weights    │             │ • Compound Scaling Architecture   │
     │ • Class-Weighted Loss Integration │             │ • Parameter-Efficient Backbone    │
     └─────────────────┬─────────────────┘             └─────────────────┬─────────────────┘
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                     ┌─────────────────────────────────────────────────────┐
                     │ Phase 4: Spatial Localization & Explainability      │
                     ├─────────────────────────────────────────────────────┤
                     │ • Gradient-Weighted Class Activation Map (Grad-CAM) │
                     │ • Feature Map Layer: resnet18.layer4 / effnet.feat │
                     │ • Visual Heatmap Overlay (ROI Extraction)           │
                     └─────────────────────────────────────────────────────┘
```

---

## Methodology & Development Phases

### Phase 1: Soft-Tissue Image Preprocessing
- **CLAHE (Contrast Limited Adaptive Histogram Equalization):** Enhances soft-tissue contrast within brain MRI slices without over-amplifying background noise (`clipLimit=2.0`, `tileGridSize=(8,8)`).
- **Bilateral Filtering:** Smooths noise while maintaining sharp tissue boundaries and lesion edges (`d=9`, `sigmaColor=75`, `sigmaSpace=75`).
- **Albumentations Augmentation:** Integrates random rotations, horizontal/vertical flips, brightness/contrast adjustments, and affine transformations to prevent overfitting.

### Phase 2: Baseline Model (ResNet-18)
- Fine-tuned pre-trained **ResNet-18** convolutional neural network.
- Custom classification head tailored to 4 diagnostic classes.
- Implemented **class-weighted cross-entropy loss** to counteract dataset class imbalance.

### Phase 3: Comparative Architecture (EfficientNet-B0)
- Benchmarked **EfficientNet-B0** against baseline ResNet-18.
- Evaluated parameter efficiency, inference latency, classification accuracy, F1-score, and confusion matrix representations under identical hyperparameter settings.

### Phase 4: Spatial Localization & Explainability (Grad-CAM)
- Generated visual heatmap overlays via **Gradient-weighted Class Activation Mapping (Grad-CAM)** on final convolutional layers (`resnet18.layer4[-1]` and `effnet.features[-1]`).
- Highlights visual region of interest (ROI) relied upon by deep neural networks during prediction.
- Conducted ablation studies evaluating the performance impact of soft-tissue preprocessing vs. raw image inputs.

---

## Benchmark Results & Experimental Evaluation

### Quantitative Performance Matrix

| Metric / Attribute | ResNet-18 (Baseline) | EfficientNet-B0 (Modern) | Architectural Advantage |
| :--- | :---: | :---: | :--- |
| **Total Parameters** | **11.18 Million** | **4.01 Million** | **64.1% parameter reduction** |
| **Validation Accuracy** | **98.57%** | **98.75%** | Higher classification accuracy |
| **Test Accuracy** | **95.62%** | **96.38%** | Superior generalization on unseen scans |
| **Macro F1-Score** | **0.956** | **0.965** | Balanced recall across all 4 classes |
| **Inference Latency** | ~14.2 ms / batch | ~9.8 ms / batch | Optimized latency for deployment |

---

## Repository Structure

```directory
e22-co543-Medical-Image-Abnormality-Cassification-Segmentation/
├── README.md                                # Root Project Documentation
├── code/
│   ├── New model/
│   │   ├── Brain_Tumor_MRI_Classification_and_Explainability.ipynb   # Main Jupyter Notebook
│   │   ├── best_resnet18.pth                # Checkpoint: Preprocessed ResNet-18 Model
│   │   ├── best_resnet18_raw.pth            # Checkpoint: Raw Image ResNet-18 Model
│   │   └── best_efficientnet_b0.pth         # Checkpoint: Fine-tuned EfficientNet-B0 Model
│   ├── Updated notebook with comparison/
│   │   ├── Brain_Tumor_MRI_Classification_and_Explainability (1).ipynb
│   │   ├── best_resnet18 (1).pth
│   │   ├── best_resnet18_pureraw.pth
│   │   └── best_resnet18_raw_aug.pth
│   └── preprocessing/
│       └── preprocessing-pipeline-visulaization.ipynb  # Preprocessing Analysis
└── docs/                                    # GitHub Pages Website & Documentation Source
    ├── _config.yml
    ├── README.md                            # Main Combined Project Web Page
    ├── data/
    │   └── index.json
    └── images/
        └── sample.png
```

---

## Environment Setup & Installation

### 1. Installation Steps

```bash
# Clone the repository
git clone https://github.com/cepdnaclk/e22-co543-Medical-Image-Abnormality-Cassification-Segmentation.git
cd e22-co543-Medical-Image-Abnormality-Cassification-Segmentation

# Create virtual environment
python -m venv venv

# Activate environment (Windows)
.\venv\Scripts\activate
# Activate environment (Linux/macOS)
source venv/bin/activate
```

### 2. Install Dependencies & Launch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python albumentations matplotlib seaborn scikit-learn pandas numpy jupyterlab

jupyter lab "code/New model/Brain_Tumor_MRI_Classification_and_Explainability.ipynb"
```

---

## Conclusion & Key Achievements

The **MedImgSys** delivers a robust computer vision infrastructure engineered for the **Medical Image Analysis Research Group at Teaching Hospital Peradeniya**. By shifting from manual visual inspections to an automated deep learning pipeline with explicit classification/segmentation architectures, the system guarantees diagnostic integrity, preserves precise spatial boundaries for lesions and anomalies, enforces consistent evaluation over sensitive data, and speeds up diagnostic report aggregation.

**Key Architecture Achievements:**
* Enhanced system architecture combining soft-tissue enhancement (CLAHE + Bilateral filtering) with state-of-the-art CNNs.
* Comprehensive coverage of classification, spatial localization, feature extraction, model optimization, and explainable diagnostic workflows.
* Parameter-efficient architecture benchmarking (**EfficientNet-B0** achieving 98.75% validation accuracy with 64% fewer parameters).
* Visual heatmaps via Grad-CAM enabling clinical-ready Automated Diagnostic Report (ADR) compilation.

---

## Links

* [Project Repository](https://github.com/cepdnaclk/{{ page.repository-name }}){:target="_blank"}
* [Project Page](https://cepdnaclk.github.io/{{ page.repository-name}}){:target="_blank"}
* [Department of Computer Engineering](http://www.ce.pdn.ac.lk/){:target="_blank"}
* [University of Peradeniya](https://eng.pdn.ac.lk/){:target="_blank"}