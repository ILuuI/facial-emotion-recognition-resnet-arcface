# Facial Emotion Recognition using Residual Networks and Angular Margin Based Discriminative Learning

Research code accompanying a manuscript under revision by Jose Carlos Arroyo Cantero, Lucas Mateo Rivadeneira Zarza, Enoc David Samur Martínez, and Wilson de Jesús Arrubla Hoyos. The study examines residual networks for classifying facial images into six expression categories: **angry, disgust, fear, happy, sad, and surprise**.

**[Reproducibility materials, supplementary tables, and selected checkpoints on Zenodo](https://doi.org/10.5281/zenodo.24238802)** · [Original notebook in this repository](FER_ResNet_ArcFace.ipynb)

## What this repository contains

This GitHub repository preserves the original `FER_ResNet_ArcFace.ipynb` notebook and the figures used to present the initial experiment. The [Zenodo release (v1.0.0)](https://doi.org/10.5281/zenodo.24238802) contains the **revised, separate notebooks for Exercises E1–E5**, two visualization/attribution notebooks, supplementary result tables, a README with input requirements, and two selected model checkpoints. Use the Zenodo notebooks when examining or rerunning the revised experiments; the original notebook here does not implement the full E1–E5 comparison.

| Location | Materials | Role |
| --- | --- | --- |
| This repository | [`FER_ResNet_ArcFace.ipynb`](FER_ResNet_ArcFace.ipynb), [`pictures/`](https://github.com/ILuuI/facial-emotion-recognition-resnet-arcface/tree/main/pictures), [`LICENSE`](LICENSE) | Historical implementation and illustrative figures. |
| [Zenodo release](https://doi.org/10.5281/zenodo.24238802) | `Exercise_1.ipynb`–`Exercise_5.ipynb`, `Feature_Visualization_ResNet56.ipynb`, `Facial_Attribution_Analysis.ipynb`, `Supplementary_Tables.xlsx`, `model_acc_94_41.pt`, `E1_best_seed5_acc_80_54.pt` | Revised experimental code, analysis notebooks, detailed results, and selected checkpoints. |

The image collections, including JLE, are **not distributed**. JLE cannot be shared under the applicable privacy and source-use restrictions. Authorized users must obtain or prepare their own permitted input data; the public files alone cannot reproduce every numerical result in the manuscript.

## Experimental design

JLE contains 39,000 images, balanced at 6,500 files per category. The revised evaluation also uses a six-class adaptation of FER2013, excluding neutral and retaining its supplied `Training`, `PublicTest`, and `PrivateTest` assignments for training, validation, and test. The five configurations were evaluated on both collections with seeds `5`, `51015`, and `543210` (30 training runs in total). Within each collection, the exercises used the same prepared partitions. Each run retained the last checkpoint and the one selected by validation accuracy; test performance was reported separately for both.

| Exercise | Configuration | Comparison |
| --- | --- | --- |
| E1 | Proposed residual network with ArcFace, scheduled angular margin and scale, and label smoothing | Main configuration. |
| E2 | Same network trained with cross-entropy | Loss function relative to E1. |
| E3 | Proposed network with fixed ArcFace margin and scale; label smoothing retained | Angular-parameter schedule relative to E1. |
| E4 | E3 without label smoothing | Label smoothing relative to E3. |
| E5 | Alternative residual architecture with the E1 ArcFace strategy | Architecture relative to E1. |

The revised runs used 50 epochs. Details of the network, schedules, preprocessing, and exported metrics are documented in the individual Zenodo notebooks. The prepared JLE test split contains real photographs and was reviewed manually for visible identity overlap with the training and validation images. This manual check does not establish a complete subject- and sequence-level audit for all sources.

## Results and interpretation

The initial experiment trained for **101 epochs** on an image-random partition of JLE and reported **94.41% test accuracy** on 3,900 images. That split could place related images or the same identity in different partitions. The value is retained as a historical result and should not be interpreted as an estimate under the revised evaluation conditions.

The values below summarize the **best-validation checkpoints** across three seeds under the revised protocol; they are test metrics, not validation metrics.

| Collection | E1 accuracy | E1 macro F1 | E2 accuracy |
| --- | ---: | ---: | ---: |
| JLE, revised test split | 80.17 ± 0.42% | 78.37 ± 0.53% | 72.84% |
| FER2013, six classes | 38.58 ± 2.85% | See supplementary tables | 68.50 ± 0.19% |

E1 exceeded E2 by **7.33 percentage points** of accuracy on JLE; E2 exceeded E1 by **29.92 points** on the six-class FER2013 evaluation. These within-collection contrasts show that the benefit of the angular-margin training strategy depends on the data and optimization conditions. They do not establish a universal advantage for ArcFace. On JLE, E1's mean recall for **fear** was 29.80%, despite its substantially higher recall for other classes. The initial 94.41% and the revised 80.17% also differ in split and training duration, so the numerical gap cannot be assigned solely to the split change.

For per-seed results, per-class recall, and the other ablations, see [`Supplementary_Tables.xlsx` on Zenodo](https://zenodo.org/records/24238802). Qualitative figures such as confusion matrices, t-SNE projections, attribution maps, and filter-activation visualizations illustrate model behavior; they do not prove identity invariance, causal explanations, or robustness by themselves.

## Using the released notebooks

Open the [Zenodo record](https://doi.org/10.5281/zenodo.24238802) and read its `README.md` before running an exercise. The notebooks target Google Colab with a CUDA-capable PyTorch environment and accept an authorized archive named `[DATASET].zip` after configuring its path. They expect six class directories under `train/`, `val/`, and `test/` (or the FER2013 aliases `Training/`, `PublicTest/`, and `PrivateTest/`, which they normalize). Run one seed and one collection at a time, keeping split membership and class indices fixed across exercises. The notebooks export run metadata, checkpoint evaluations, histories, and per-class results.

The two released `.pt` files represent **different protocols**: `model_acc_94_41.pt` is associated with the initial image-random experiment, while `E1_best_seed5_acc_80_54.pt` is the E1 checkpoint selected by validation accuracy in the revised JLE evaluation (seed 5). The number in the latter filename denotes its reported test accuracy; test results were not used to select the checkpoint.

## Scope, citation, and rights

These models classify **facial-image expression labels**. Their predictions do not determine a person's internal emotional or psychological state, and the study does not establish clinical diagnostic validity. The data include different image types, including posed, synthetic, and illustrative faces; performance can vary by source and class.

The notebook code in this GitHub repository is provided under the [MIT License](LICENSE). The Zenodo record identifies MIT for its notebook code and CC BY 4.0 for its README, supplementary tables, and selected model weights; consult its file-level license statement. Neither license grants rights to third-party datasets or private images absent from the release. If you use the released materials, cite the **[Zenodo version DOI](https://doi.org/10.5281/zenodo.24238802)**. The manuscript is under revision and does not yet have a final publication DOI.
