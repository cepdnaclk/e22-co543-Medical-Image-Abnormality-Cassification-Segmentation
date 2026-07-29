---

layout: home
permalink: index.html

## repository-name: e22-co543-Medical-Image-Abnormality-Cassification-Segmentation
title: Medical Image Abnormality Classification & Segmentation

# Medical Image Abnormality Classification and Segmentation System

---


* E/22/130, Hansara S. H. S., [e22130@eng.pdn.ac.lk](https://www.google.com/search?q=mailto%3Ae22130%40eng.pdn.ac.lk)
* E/22/211, H.M. Liyanage, [e22211@eng.pdn.ac.lk](https://www.google.com/search?q=mailto%3Ae22211%40eng.pdn.ac.lk)
* E/22/044, D.M.N.N. Bandara, [e22044@eng.pdn.ac.lk](https://www.google.com/search?q=mailto%3Ae22044%40eng.pdn.ac.lk)
* E/22/421, Weerasinghe W. P. T. H., [e22211@eng.pdn.ac.lk](https://www.google.com/search?q=mailto%3Ae22421%40eng.pdn.ac.lk)

#### Table of Contents

1. [Introduction](https://www.google.com/search?q=%23introduction)
2. [System Architecture & Conceptual Design](https://www.google.com/search?q=%23system-architecture--conceptual-design)
3. [Conclusion](https://www.google.com/search?q=%23conclusion)
4. [Links](https://www.google.com/search?q=%23links)

---

## Introduction

The Medical Image Analysis Research Group at the Teaching Hospital Peradeniya handles critical diagnostic imaging cases (e.g., tumors, fractures, internal anomalies) and radiographic MRI/CT examinations. Currently, the department relies on manual visual inspection, physical radiological films, and human-based Diagnostic Imaging Reports (DIR).

This manual workflow creates critical operational bottlenecks:

* **Inefficient Anomaly Detection:** Manual visual inspection makes tracing subtle historical patient anomalies or matching microscopic patterns to physical films extremely slow.
* **Diagnostic Accuracy Vulnerabilities:** Tracking complex biological structures, tumor boundaries, lesions, and laboratory features across manual inspections risks misinterpretation or diagnostic gaps.
* **Data Processing & Granular Image Segmentation:** Sensitive radiographic findings, patient anomaly details, and clinical scans lack strict automated pixel-level boundary restrictions.
* **Report Compilation Delay:** Assembling Automated Diagnostic Reports (ADR) for clinical submissions requires manual transcription from multiple unlinked imaging modalities.

**MedImgSys** is a deep learning system engineered specifically to automate, secure, and streamline the diagnostic imaging workflows of the Radiology Department. The primary objective of this project is to implement a robust, convolutional neural network pipeline that guarantees high accuracy, enforces precise segmentation constraints, maintains performance logging via metric tracking, and optimizes complex model architectures for diagnostic report compilation.

---

## System Architecture & Conceptual Design

The core of this system is designed around strict machine learning principles to ensure minimal false negatives, high diagnostic integrity, explicit feature extraction via specialized network architectures, and structured model optimization.

### System Architecture Breakdown



---

## Conclusion

The **MedImgSys** delivers a computer vision infrastructure engineered for the Medical Image Analysis Research Group at Teaching Hospital Peradeniya. By shifting from manual visual inspections to an automated deep learning pipeline with explicit classification/segmentation architectures, the system guarantees diagnostic integrity, preserves precise spatial boundaries for lesions and anomalies, enforces consistent evaluation over sensitive data, and speeds up diagnostic report aggregation.

Key Architecture Achievements:

* An Enhanced System Architecture utilizing specialization/generalization hierarchies (`Dataset` and `Model_Table` superclasses).
* Complete coverage of classification, segmentation, feature extraction, optimization, and automated diagnostic workflows.
* Strict enforcement of high-accuracy metrics, spatial boundary constraints, robust model checkpoints, and training logging.
* Structured output visualizations enabling efficient clinical-ready Automated Diagnostic Report (ADR) generation.

---

## Links

* [Project Repository](https://github.com/cepdnaclk/{{ page.repository-name }}){:target="_blank"}
* [Project Page](https://cepdnaclk.github.io/{{ page.repository-name}}){:target="_blank"}
* [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
* [University of Peradeniya](https://eng.pdn.ac.lk/)