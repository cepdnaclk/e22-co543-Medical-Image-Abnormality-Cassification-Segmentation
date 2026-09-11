# Brain Tumor MRI Classification, Medical Preprocessing Engine & Visual Explainability

<div align="center">

![Project Banner](docs/images/banner.svg)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA Acceleration](https://img.shields.io/badge/NVIDIA%20CUDA-13.0%20RTX%206000%20Ada-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Albumentations](https://img.shields.io/badge/Albumentations-2.0+-FF4F00?style=for-the-badge)](https://albumentations.ai/)
[![Test Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.69%25-brightgreen?style=for-the-badge)](https://github.com/cepdnaclk/e22-co543-Medical-Image-Abnormality-Cassification-Segmentation)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**Course:** CO5430 Computer Vision | **Group:** G14 | **Project ID:** P19  
**Institution:** Department of Computer Engineering, Faculty of Engineering, University of Peradeniya  
**Clinical Context:** Medical Image Analysis Research Group, Teaching Hospital Peradeniya

[Overview](#-project-overview) • [System Architecture](#-system-architecture) • [Dataset](#-dataset--stratified-splitting) • [Preprocessing Engine](#-medical-image-preprocessing-engine) • [Model Architectures](#-model-architectures--training-pipeline) • [Benchmarking Results](#-quantitative-model-benchmarking) • [Ablation Study](#-3-stage-ablation-study) • [Grad-CAM Explainability](#-spatial-localization--explainability-grad-cam) • [Quickstart](#-getting-started--reproducibility) • [Team](#-team--contributors)

</div>

---

## 📌 Project Overview

In clinical neuro-oncology, magnetic resonance imaging (MRI) is the gold standard for detecting and categorizing intracranial neoplasms. However, standard clinical radiology workflows at regional healthcare centers—such as the **Teaching Hospital Peradeniya**—often face severe operational bottlenecks:
- **Diagnostic Delay:** Manual inspection of multi-slice radiographic films and manual transcription into Diagnostic Imaging Reports (DIR) creates significant turnaround delays.
- **Inter-Observer Variability:** Subtle morphological variations and boundary ambiguities between high-grade **gliomas**, extra-axial **meningiomas**, and intrasellar **pituitary adenomas** can lead to diagnostic discrepancies.
- **Black-Box AI Skepticism:** Standard deep convolutional networks offer high statistical accuracy but lack spatial interpretability, creating a barrier to clinical adoption by radiologists and neurosurgeons.

**MedImgSys** is an end-to-end deep learning and computer vision framework designed to automate brain tumor classification, enforce rigid anatomical preprocessing, and provide **zero-annotation spatial localization** through visual explainability.

### 🌟 Key Highlights
- **95.69% Official Test Accuracy** achieved on a balanced evaluation benchmark of 1,600 unseen MRI scans.
- **Medical Image Preprocessing Engine:** Implements Otsu contour skull-stripping, edge-preserving bilateral filtering, and LAB-space Contrast Limited Adaptive Histogram Equalization (CLAHE).
- **+22.50% Preprocessing Boost:** Demonstrates through a controlled 3-stage ablation study that custom medical preprocessing elevates performance from **73.19%** (pure raw) to **95.69%** (full pipeline).
- **Architectural Parity with 79% Fewer FLOPs:** Benchmarks **ResNet-18** against **EfficientNet-B0**, showing that compound scaling achieves statistical parity (95.62% vs 95.69%) while reducing parameter overhead by **64.1%** and computational FLOPs by **78.5%**.
- **Grad-CAM Visual Explainability:** Derives localized saliency heatmaps overlaid on anatomical scans, validating that the models focus strictly on tumor pathology rather than acquisition artifacts.

---

## 🏗 System Architecture

The pipeline follows a modular, reproducible design spanning raw data ingestion, automated preprocessing, data augmentation, dual-backbone modeling, quantitative evaluation, and visual localization:

<div align="center">

![End-to-End System Architecture](docs/images/system_architecture.svg)

</div>

---

## 📊 Dataset & Stratified Splitting

The pipeline uses the public **Kaggle Brain Tumor MRI Dataset** (`masoudnickparvar/brain-tumor-mri-dataset`), comprising **7,022 total T1-weighted contrast-enhanced MRI scans** categorized into four clinically distinct classes:

| Class | Anatomical Description | Clinical Significance |
| :--- | :--- | :--- |
| **Glioma** | Infiltrative intra-axial tumor originating from glial cells | Highly invasive; diffuse subcortical borders requiring aggressive resection |
| **Meningioma** | Extra-axial, slow-growing tumor arising from the meningeal dural layer | Sharp dural tail; extra-axial mass displacement of cortical structures |
| **Pituitary Adenoma** | Neoplasm originating in the anterior pituitary gland within the sella turcica | Endocrine disruption and visual field deficits via optic chiasm compression |
| **No Tumor** | Healthy cerebral tissue without structural abnormalities | Baseline negative control essential for minimizing false positive interventions |

### Partitioning & Balance

To eliminate data leakage and avoid non-stratified source distributions, the dataset is partitioned using a fixed global seed (`42`):
- **Official Test Set:** **1,600 images** (Strictly balanced: **400 images per class**; 25.0% each)
- **Training Set:** **4,480 images** (Stratified 80% split from training pool: 1,320 Glioma, 1,339 Meningioma, 1,595 No Tumor, 1,456 Pituitary)
- **Validation Set:** **1,120 images** (Stratified 20% split from training pool: 330 Glioma, 335 Meningioma, 399 No Tumor, 364 Pituitary)
- **Class Weights:** Dynamic inverse-frequency weighting applied during Cross-Entropy optimization:
$$\text{Weight}_c = \frac{N_{\text{total}}}{C \cdot N_c}$$

---

## 🔬 Medical Image Preprocessing Engine

Raw MRI scans often contain non-standardized padding, acquisition noise from magnetic coil variance, and low contrast across soft-tissue boundaries. We implement a sequential 4-stage OpenCV preprocessing engine:

<div align="center">

![Medical Preprocessing Workflow](docs/images/preprocessing_workflow.svg)

</div>

### Algorithmic Breakdown

```
Raw MRI Scan (Heterogeneous Size / Black Margins)
       │
       ▼
[Stage 1] Skull Stripping & Bounding Box Contour Crop
       │  • Grayscale conversion -> Gaussian blur (5×5, σ=0)
       │  • Otsu automatic thresholding (thresh=45, max=255)
       │  • Morphological Erode & Dilate (2 iterations, 3×3 kernel)
       │  • Largest external contour detection (contourArea > 500)
       │  • Bounding box rectangle crop (removes 40-60% black background)
       ▼
[Stage 2] Edge-Preserving Bilateral Filtering
       │  • cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
       │  • Replaces pixels with weighted Gaussian distance & photometric intensity
       │  • Eliminates high-frequency MRI noise while preserving tumor edges
       ▼
[Stage 3] Contrast Limited Adaptive Histogram Equalization (CLAHE)
       │  • BGR -> LAB Color Space conversion
       │  • CLAHE applied exclusively to L (Lightness) channel (clipLimit=2.0, tileGrid=(8,8))
       │  • Local histogram equalization with noise clip-limiting
       │  • Merge with original A & B chroma channels -> convert to BGR
       ▼
[Stage 4] Standardization & Tensor Rescaling
          • cv2.INTER_AREA interpolation resize to 224 × 224 × 3
          • Channel reordering BGR -> RGB
          • Normalization via ImageNet distribution: μ = [0.485, 0.456, 0.406], σ = [0.229, 0.224, 0.225]
```

---

## 🔄 Data Augmentations (Albumentations Suite)

To prevent overfitting on limited slice angles and scanner types, training batches pass through stochastic augmentations using `albumentations`:

| Transform | Parameters | Clinical Rationale |
| :--- | :--- | :--- |
| **Horizontal Flip** | $p = 0.5$ | Simulates hemispheric symmetry of cerebral anatomy |
| **Vertical Flip** | $p = 0.3$ | Simulates variable orientation on sagittal/coronal acquisition |
| **ShiftScaleRotate** | Shift $\pm 8\%$, Scale $\pm 10\%$, Angle $\pm 25^\circ$, $p = 0.7$ | Accounts for slight patient head misalignment in MRI coils |
| **ColorJitter** | Brightness $\pm 0.2$, Contrast $\pm 0.2$, $p = 0.5$ | Replicates magnetic field heterogeneity across 1.5T and 3.0T scanners |
| **GaussianBlur** | Kernel $(3, 5)$, $p = 0.2$ | Simulates slight patient movement / phase-encoding artifacts |
| **CoarseDropout** | Max holes $= 6$, Max size $= 16 \times 16$, $p = 0.3$ | Regularizes deep feature representations; mimics focal dropouts |

---

## 🧠 Model Architectures & Training Pipeline

We train and evaluate two complementary convolutional deep learning paradigms:

### 1. ResNet-18 (Residual Learning Baseline)
- Incorporates residual bottleneck skip connections:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$
- Mitigates vanishing gradients across deep feature spaces while maintaining direct gradient flow.
- Parameter Count: **11.18 Million** | Computational Complexity: **1.81 GFLOPs**.

### 2. EfficientNet-B0 (Compound-Scaled Modern Benchmark)
- Utilizes inverted bottleneck convolutions (**MBConv**) with depthwise separable convolutions and Squeeze-and-Excitation (**SE**) attention blocks:
$$\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi$$
- Balances network depth, channel width, and feature resolution systematically.
- Parameter Count: **4.01 Million (-64.1% reduction)** | Complexity: **0.39 GFLOPs (-78.5% reduction)**.

### Optimization Protocol
- **Optimizer:** `AdamW` (`lr` = $3 \times 10^{-4}$, `weight_decay` = $1 \times 10^{-4}$)
- **Learning Rate Scheduler:** `CosineAnnealingLR` ($T_{\max} = 10$, $\eta_{\min} = 1 \times 10^{-6}$)
- **Hardware Platform:** NVIDIA RTX 6000 Ada Generation
- **Acceleration:** PyTorch Automatic Mixed Precision (`torch.cuda.amp.autocast()` and `GradScaler()`)
- **Batch Size:** 32 | **Epochs:** 10

---

## 📈 Quantitative Model Benchmarking

Both architectures were benchmarked against the **1,600 unseen scans** of the official test set:

<div align="center">

![Model Benchmark Comparison](docs/images/model_benchmark_comparison.svg)

</div>

### Performance Summary Table

| Model Architecture | Test Accuracy (%) | Macro Precision | Macro Recall | Macro F1-Score | Parameter Count | Est. Complexity | Latency (ms/scan) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-18 (Baseline)** | **95.69%** | **0.9588** | **0.9569** | **0.9563** | 11.18 M | 1.81 GFLOPs | **5.04 ms** |
| **EfficientNet-B0 (Modern)** | **95.62%** | **0.9587** | **0.9562** | **0.9555** | **4.01 M** | **0.39 GFLOPs** | **5.40 ms** |

### Key Findings
1. **Accuracy Parity:** EfficientNet-B0 achieves virtually identical test classification accuracy (**95.62%** vs. **95.69%**) and macro F1-score (**0.9555** vs. **0.9563**) compared to ResNet-18.
2. **Extreme Resource Efficiency:** EfficientNet-B0 reduces model parameter size by **64.1%** (4.01M vs 11.18M) and cuts computational floating-point operations by **78.5%** (0.39 vs 1.81 GFLOPs).
3. **Deployment Recommendation:** For cloud-based PACS servers with high throughput, ResNet-18 provides the fastest raw inference latency (5.04 ms/img). For edge-embedded clinical MRI workstations, mobile radiological devices, and memory-constrained deployments, EfficientNet-B0 is the ideal choice.

---

## 🧪 3-Stage Ablation Study

To isolate the exact contribution of the medical preprocessing engine versus data augmentation, a controlled 3-stage ablation study was conducted with ResNet-18:

<div align="center">

![Ablation Study Chart](docs/images/ablation_study_chart.svg)

</div>

### Ablation Experimental Results

| Stage | Configuration Description | Preprocessing Applied | Augmentations Applied | Test Accuracy | Improvement |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | **Pure Raw Baseline** | ❌ None (Raw MRI) | ❌ None | **73.19%** | Baseline |
| **2** | **Raw Scans + Augmentations** | ❌ None (Raw MRI) | ✅ Albumentations | **93.31%** | **+20.12%** |
| **3** | **Full Medical Pipeline** | ✅ Contour Crop + Bilateral + CLAHE | ✅ Albumentations | **95.69%** | **+22.50%** |

### Ablation Insights
- **The Geometry Gap (+20.12%):** Training on raw scans without augmentation leads to severe memorization of scanner noise, achieving only 73.19%. Adding geometric and photometric augmentation forces the network to learn invariant features, boosting accuracy to 93.31%.
- **The Medical Contrast Boost (+2.38% / +22.50% total):** Combining skull contour stripping, bilateral filtering, and LAB CLAHE elevates performance to **95.69%**. Preprocessing strips irrelevant background pixels and sharpens subtle soft-tissue boundaries, enabling the model to disambiguate borderline meningiomas from infiltrative gliomas.

---

## 🔍 Spatial Localization & Explainability (Grad-CAM)

To provide verifiable spatial interpretability without manual pixel-level segmentation masks, we implement **Grad-CAM (Gradient-Weighted Class Activation Mapping)**:

<div align="center">

![Grad-CAM Explainability Workflow](docs/images/gradcam_workflow.svg)

</div>

### Mathematical Formulation

1. **Gradient Computation:** Compute the gradient of the predicted class score $y^c$ with respect to feature activation map $A^k$ of the final convolutional layer:
$$\frac{\partial y^c}{\partial A^k}$$

2. **Neuron Importance Weights ($\alpha_k^c$):** Apply global average pooling over spatial dimensions $(i, j)$ of height $U$ and width $V$:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A_{i,j}^k}$$

3. **Linear Combination and Rectification:** Compute the weighted sum of forward feature activation maps, followed by a ReLU operation to preserve features that contribute positively to the class:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

4. **Heatmap Blending:** Normalize $L_{\text{Grad-CAM}}^c$ to $[0, 1]$, upsample to $224 \times 224$, apply `cv2.COLORMAP_JET`, and blend with the preprocessed anatomical MRI scan at $\alpha = 0.4$:
$$I_{\text{overlay}} = 0.4 \cdot I_{\text{heatmap}} + 0.6 \cdot I_{\text{anatomical}}$$

### Clinical Validation Across Classes
- **Glioma:** Activation maps localize deeply within intra-axial, subcortical white matter regions, capturing the irregular, infiltrative margins typical of astrocytic tumors.
- **Meningioma:** Activation concentrates along extra-axial dural interfaces and peripheral convexities, pinpointing the characteristic dural tail enhancement.
- **Pituitary Adenoma:** Heatmaps lock precisely onto the sellar and suprasellar regions at the skull base, correctly highlighting micro- and macroadenomas.
- **No Tumor:** Visualizations display low-magnitude, diffuse, non-focal activations across normal cerebral parenchyma, confirming the absence of pathological focal points.

---

## 📂 Repository Structure

```
e22-co543-Medical-Image-Abnormality-Cassification-Segmentation/
├── README.md                                          # Master repository documentation (this file)
├── code/
│   ├── New model/
│   │   ├── Brain_Tumor_MRI_Classification_and_Explainability.ipynb
│   │   ├── best_resnet18.pth                          # Weights: ResNet-18 (Full Pipeline)
│   │   ├── best_resnet18_raw.pth                      # Weights: ResNet-18 (Raw Scans)
│   │   └── best_efficientnet_b0.pth                   # Weights: EfficientNet-B0 (Full Pipeline)
│   ├── Updated notebook with comparison/
│   │   ├── Brain_Tumor_MRI_Classification_and_Explainability (1).ipynb # Full end-to-end benchmark
│   │   ├── best_resnet18 (1).pth
│   │   ├── best_efficientnet_b0 (1).pth
│   │   ├── best_resnet18_pureraw.pth                  # Weights: Ablation Stage 1 (Pure Raw)
│   │   └── best_resnet18_raw_aug.pth                  # Weights: Ablation Stage 2 (Raw + Aug)
│   └── preprocessing/
│       └── preprocessing-pipeline-visulaization.ipynb # Data engineering & 5-stage pipeline
└── docs/
    ├── _config.yml                                    # Jekyll GitHub Pages configuration
    ├── README.md                                      # Documentation site page
    ├── data/
    │   └── index.json                                 # Project team metadata & tags
    └── images/
        ├── banner.svg                                 # High-resolution vector header banner
        ├── system_architecture.svg                    # Full architecture diagram
        ├── preprocessing_workflow.svg                 # 4-stage preprocessing engine flowchart
        ├── model_benchmark_comparison.svg             # ResNet-18 vs EfficientNet-B0 benchmark
        ├── ablation_study_chart.svg                   # 3-stage ablation study comparison
        ├── gradcam_workflow.svg                       # Grad-CAM localization & explainability
        └── sample.png                                 # Repository asset
```

---

## 🚀 Getting Started & Reproducibility

### Prerequisites
- Python 3.10, 3.11, or 3.12
- NVIDIA GPU with CUDA 11.8+ or 12.0+ (Tested on NVIDIA RTX 6000 Ada Generation)
- Minimum 8 GB GPU VRAM recommended (CPU execution supported via fallback)

### 1. Clone the Repository
```bash
git clone https://github.com/cepdnaclk/e22-co543-Medical-Image-Abnormality-Cassification-Segmentation.git
cd e22-co543-Medical-Image-Abnormality-Cassification-Segmentation
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
# On Linux / macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install core deep learning & computer vision libraries:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install albumentations opencv-python numpy pandas matplotlib seaborn scikit-learn timm torchinfo thop
```

### 3. Dataset Configuration
Download the [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and place it into an `archive/` folder in the project root:
```
archive/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```
The notebook automatically detects `archive/`, `Archive/`, or parent directory paths across Windows and Linux environments.

### 4. Running the Pipeline
Launch Jupyter Lab or Notebook:
```bash
jupyter lab
```
Open and execute:
```
code/Updated notebook with comparison/Brain_Tumor_MRI_Classification_and_Explainability (1).ipynb
```
The notebook will:
1. Detect CUDA hardware capabilities and initialize seed `42`.
2. Perform stratified 80/20 train-validation splitting.
3. Apply the 4-stage preprocessing engine and Albumentations.
4. Train ResNet-18 and EfficientNet-B0 with mixed precision.
5. Generate confusion matrices, ROC curves, and quantitative benchmark tables.
6. Execute the 3-stage ablation study.
7. Compute and render Grad-CAM explainability heatmaps for all 4 classes.

---

## 👥 Team & Contributors

This project was developed by **Group 14** as part of the **CO5430 Computer Vision** course at the **Department of Computer Engineering, University of Peradeniya**:

<table align="center">
  <tr>
    <td align="center" width="25%">
      <img src="https://people.ce.pdn.ac.lk/images/students/e22/e22130.jpg" alt="Hansara S. H. S." width="140" height="140" style="border-radius: 50%; object-fit: cover;"/><br />
      <b>Hansara S. H. S.</b><br />
      E/22/130<br />
      <a href="mailto:e22130@eng.pdn.ac.lk">e22130@eng.pdn.ac.lk</a>
    </td>
    <td align="center" width="25%">
      <img src="https://people.ce.pdn.ac.lk/images/students/e22/e22211.jpg" alt="H. M. Liyanage" width="140" height="140" style="border-radius: 50%; object-fit: cover;"/><br />
      <b>H. M. Liyanage</b><br />
      E/22/211<br />
      <a href="mailto:e22211@eng.pdn.ac.lk">e22211@eng.pdn.ac.lk</a>
    </td>
    <td align="center" width="25%">
      <img src="https://people.ce.pdn.ac.lk/images/students/e22/e22044.jpg" alt="D. M. N. N. Bandara" width="140" height="140" style="border-radius: 50%; object-fit: cover;"/><br />
      <b>D. M. N. N. Bandara</b><br />
      E/22/044<br />
      <a href="mailto:e22044@eng.pdn.ac.lk">e22044@eng.pdn.ac.lk</a>
    </td>
    <td align="center" width="25%">
      <img src="https://people.ce.pdn.ac.lk/images/students/e22/e22421.jpg" alt="Weerasinghe W. P. T. H." width="140" height="140" style="border-radius: 50%; object-fit: cover;"/><br />
      <b>Weerasinghe W. P. T. H.</b><br />
      E/22/421<br />
      <a href="mailto:e22421@eng.pdn.ac.lk">e22421@eng.pdn.ac.lk</a>
    </td>
  </tr>
</table>

---

## 🔗 Project Links

- **GitHub Repository:** [cepdnaclk/e22-co543-Medical-Image-Abnormality-Cassification-Segmentation](https://github.com/cepdnaclk/e22-co543-Medical-Image-Abnormality-Cassification-Segmentation)
- **Project Documentation Page:** [cepdnaclk.github.io/e22-co543-Medical-Image-Abnormality-Cassification-Segmentation](https://cepdnaclk.github.io/e22-co543-Medical-Image-Abnormality-Cassification-Segmentation/)
- **Department of Computer Engineering:** [ce.pdn.ac.lk](http://www.ce.pdn.ac.lk/)
- **Faculty of Engineering, University of Peradeniya:** [eng.pdn.ac.lk](https://eng.pdn.ac.lk/)

---

## 📜 References & Acknowledgements

1. **Selvaraju, R. R., et al.** (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. IEEE International Conference on Computer Vision (ICCV).
2. **He, K., et al.** (2016). *Deep Residual Learning for Image Recognition*. IEEE Conference on Computer Vision and Pattern Recognition (CVPR).
3. **Tan, M., & Le, Q. V.** (2019). *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*. International Conference on Machine Learning (ICML).
4. **Zuiderveld, K.** (1994). *Contrast Limited Adaptive Histogram Equalization*. Graphics Gems IV, Academic Press Professional, Inc.
5. **Tomasi, C., & Manduchi, R.** (1998). *Bilateral Filtering for Gray and Color Images*. IEEE International Conference on Computer Vision (ICCV).
6. **Kaggle MRI Dataset:** [Brain Tumor MRI Dataset by Masoud Nickparvar](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).