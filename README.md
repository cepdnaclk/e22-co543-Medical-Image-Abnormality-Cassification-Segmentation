# Brain Tumor MRI Classification & Localization

**Course:** CO5430 Computer Vision | **Group:** G14 | **Project ID:** P19[cite: 3]
**Tags:** Image Processing, Web

## Team Members
* **Hansara S. H. S.** (E/22/130) - [e22130@eng.pdn.ac.lk](mailto:e22130@eng.pdn.ac.lk)
* **H.M. Liyanage** (E/22/211) - [e22211@eng.pdn.ac.lk](mailto:e22211@eng.pdn.ac.lk)
* **D.M.N.N. Bandara** (E/22/044) - [e22044@eng.pdn.ac.lk](mailto:e22044@eng.pdn.ac.lk)
* **Weerasinghe W. P. T. H.** (E/22/421) - [e22421@eng.pdn.ac.lk](mailto:e22421@eng.pdn.ac.lk)

## Supervisors
* **Dr. Supervisor 1** - [email@eng.pdn.ac.lk](mailto:email@eng.pdn.ac.lk)
* **Supervisor 2** - [email@eng.pdn.ac.lk](mailto:email@eng.pdn.ac.lk)

## Project Overview
This project focuses on combining image classification with spatial localization to highlight pathological brain tumor regions[cite: 3]. The development is structured across four progressive phases[cite: 3]:

* **Phase 1: Dataset Setup & Preprocessing** - Implements a strict 80/10/10 stratified split, CLAHE for soft-tissue enhancement, bilateral filtering, and Albumentations augmentations[cite: 3].
* **Phase 2: Baseline Model** - Fine-tunes a pre-trained ResNet-18 CNN with class-weighted loss functions to mitigate dataset imbalance[cite: 3].
* **Phase 3: Comparative Architecture** - Integrates and benchmarks an EfficientNet-B0 model to evaluate gains over the baseline[cite: 3].
* **Phase 4: Spatial Localization & Explainability** - Utilizes Grad-CAM to generate visual heatmap overlays of abnormal regions and includes an ablation study on preprocessing impacts[cite: 3].

## Dataset
* **Source:** Kaggle Brain Tumor MRI Dataset[cite: 3]
* **Size:** 7,022 Scans[cite: 3]
* **Classes:** Glioma, Meningioma, Pituitary, No Tumor[cite: 3]