# Facial Emotion Recognition using Residual Networks and Angular Margin Based Discriminative Learning

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Google Colab](https://img.shields.io/badge/Run%20in-Colab-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/)

**Universidad de Sucre · Sincelejo-Sucre, Colombia**

[Paper](#citation) · [Quick Start](#quick-start) · [Results](#results) · [Architecture](#architecture)

</div>

---

## Overview

This repository contains the official implementation of our paper:

> **Facial Emotion Recognition using Residual Networks and Angular Margin Based Discriminative Learning**
> Jose Carlos Arroyo Cantero, Lucas Mateo Rivadeneira Zarza, Enoc David Samur Martinez, Wilson de Jesús Arrubla Hoyos
> *INGE CUC · Universidad de Sucre, Colombia*
> [Read the full paper](#) ← *(add URL when published)*

Deep learning has significantly advanced facial emotion recognition; however, achieving robust performance across heterogeneous datasets remains a challenging task due to variations in image quality, acquisition conditions, and class distributions. This paper presents a deep residual architecture for facial emotion recognition trained with **ArcFace Loss**, a metric learning approach based on angular margins that enhances feature discrimination among emotion categories. The proposed model is trained on a curated dataset of **39,000 facial images** collected from **33 heterogeneous sources**, including standardized public benchmarks, real-world photographs, AI-generated images, and manually labeled subsets balanced across all six universal emotions. Extensive experiments show that the proposed model achieves robust and reliable performance, reaching **94.41% accuracy** and a **macro F1-score of 0.9441** on a held-out test set of **3,900 images**.


---

## Emotions Classified

| Class | Emotion |
|-------|---------|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |

---

## Architecture

The model is built on a custom **Residual Network (ResNet)** with the following design choices:

- **Residual blocks** with 3×3 convolutions; channels scale progressively from 32 → 256
- **Activation functions**: ReLU and SiLU for modeling nonlinear relationships
- **Spatial downsampling** via convolutions with stride 2 (128×128 → 8×8)
- **Classification head**: Multilayer Perceptron (MLP) over extracted embeddings
- **Loss function**: ArcFace Loss with angular margin `m` for intra-class compactness and inter-class separation

The residual connection follows the standard formulation:

$$H(x) = F(x) + x$$

And the ArcFace loss is defined as:

$$\mathcal{L} = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j=1,\, j \neq y_i}^{N} e^{s \cdot \cos(\theta_j)}}$$

---

## Results

### Overall Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | **94.41%** |
| Macro Precision | 0.9442 |
| Macro Recall | 0.9441 |
| Macro F1-Score | 0.9441 |

### Performance by Emotion Class

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Angry | 0.9437 | 0.9277 | 0.9356 | 650 |
| Disgust | 0.9254 | 0.9354 | 0.9304 | 650 |
| Fear | 0.9560 | 0.9354 | 0.9456 | 650 |
| **Happy** | **0.9756** | **0.9862** | **0.9809** | 650 |
| Sad | 0.9311 | 0.9354 | 0.9332 | 650 |
| Surprise | 0.9331 | 0.9446 | 0.9388 | 650 |

> The **Happy** class achieved the highest F1-score (0.9809), while **Disgust** and **Sad** presented the greatest classification challenge, likely due to partial overlap in their facial action units.

---

## Repository Structure

```
facial-emotion-recognition-resnet-arcface/
│
├── README.md
├── LICENSE                    # MIT License
├── requirements.txt

```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/facial-emotion-recognition-resnet-arcface.git
cd facial-emotion-recognition-resnet-arcface
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python src/train.py --config configs/config.yaml
```

### 4. Evaluate

```bash
python src/evaluate.py --weights path/to/checkpoint.pth
```

### 5. Run in Google Colab

Open the training notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) ← *(update with your notebook link)*

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | SGD (momentum=0.9, Nesterov) |
| Weight Decay | 1×10⁻⁴ |
| Epochs | 101 |
| Batch Size | 160 |
| Max Learning Rate | 0.05 |
| LR Scheduler | OneCycleLR (cosine) |
| Mixed Precision | FP16 |
| Global Seed | 20000807 |
| Input Size | 128×128 (grayscale) |
| ArcFace Margin `m` | Scheduled (warmup) |
| Label Smoothing ε | 0.05 → 0.02 → 0 |
| Drop Path | up to 0.05 |

---

## Dataset

The training dataset consists of **39,000 facial images** organized into **6 balanced emotional classes**. It was compiled from:

- Standardized public datasets widely used in the scientific literature (see Appendix in the paper)
- AI-generated images
- Photographs captured under real-world conditions
- Images from open public sources with manual labeling

**Split:**

| Subset | Proportion | Samples |
|--------|-----------|---------|
| Training | 80% | ~31,200 |
| Validation | 10% | ~3,900 |
| Test | 10% | 3,900 |

> **Note:** Due to the aggregated nature of the dataset, strict identity-level partitioning was not enforced. This is acknowledged as a limitation and proposed as future work.

---

## Interpretability

We evaluated multiple visual explainability techniques including Grad-CAM, EigenCAM, SmoothGrad, Integrated Gradients, Blur Occlusion, and VarGrad.

**VarGrad** was adopted as the primary method due to its robustness with ArcFace-based architectures. By estimating gradient variance under stochastic perturbations, it produces more accurate and spatially consistent attribution maps — particularly important since ArcFace optimizes angular distances in embedding space rather than conventional classification logits.

---

## Authors

| Name | Institution | Email | ORCID |
|------|-------------|-------|-------|
| Jose Carlos Arroyo Cantero | Universidad de Sucre | jose.arroyo54@unisucrevirtual.edu.co | [0009-0009-9449-8105](https://orcid.org/0009-0009-9449-8105) |
| Lucas Mateo Rivadeneira Zarza | Universidad de Sucre | lucas.rivadeneira@unisucrevirtual.edu.co | [0009-0004-7201-2905](https://orcid.org/0009-0004-7201-2905) |
| Enoc David Samur Martinez | Universidad de Sucre | enoc.samur@unisucrevirtual.edu.co | [0009-0002-7579-4944](https://orcid.org/0009-0002-7579-4944) |
| Wilson de Jesús Arrubla Hoyos | Universidad de Sucre | wilson.arrubla@unisucrevirtual.edu.co | [0000-0001-7119-7603](https://orcid.org/0000-0001-7119-7603) |

### CRediT — Contributor Roles Taxonomy

| Contributor | Roles |
|-------------|-------|
| **Jose C. Arroyo** | Data Curation · Investigation · Methodology · Project Administration · Software · Visualization · Validation · Writing – Original Draft · Writing – Review & Editing |
| **Lucas M. Rivadeneira** | Data Curation · Investigation · Methodology · Software · Visualization · Writing – Original Draft · Writing – Review & Editing |
| **Enoc D. Samur** | Data Curation · Investigation · Methodology · Software · Visualization · Writing – Original Draft · Writing – Review & Editing |
| **Wilson J. Arrubla** | Supervision |

---

## Citation

If you use this work in your research, please cite:

```bibtex
@article{arroyo2026fer,
  title     = {Facial Emotion Recognition using Residual Networks and Angular Margin Based Discriminative Learning},
  author    = {Arroyo Cantero, Jose Carlos and Rivadeneira Zarza, Lucas Mateo and Samur Martinez, Enoc David and Arrubla Hoyos, Wilson de Jes{\'u}s},
  journal   = {INGE CUC},
  year      = {2026},
  publisher = {Universidad de Sucre},
  url       = {}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## References

Key references from the paper:

- Ekman, P. — Universal basic emotions and facial action coding (FACS)
- ArcFace: Additive Angular Margin Loss for Deep Face Recognition (Deng et al., 2019)
- ResNet: Deep Residual Learning for Image Recognition (He et al., 2016)
- PyTorch: An Imperative Style, High-Performance Deep Learning Library
- AffectNet: A Database for Facial Expression, Valence, and Arousal in the Wild

---

<div align="center">
Made with ❤️ by <strong>PEPA PIC Research Group</strong> · Universidad de Sucre · Sincelejo, Colombia
</div>
