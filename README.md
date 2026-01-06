# ☀️ SunGuard: Severe Solar Flare Classification (SDO Multi-View Fusion) ☀️

[![Paper](https://img.shields.io/badge/paper-PDF-red)](./SunGuard_Paper_Graduation_Project.pdf)
[![Dataset](https://img.shields.io/badge/dataset-SDOBenchmark-blue)](http://i4ds.github.io/SDOBenchmark/)

> **SunGuard** is a modular **multi-view** framework for **severe (≥ M-class)** solar flare classification using **10 synchronized SDO channels** (AIA + HMI).  
> We learn **wavelength-specific embeddings** using **EfficientNetV2-S**, then fuse them using either a **1D Residual Neural Fusion (ResNet)** model or **XGBoost**.  
> We handle extreme imbalance (~1:15) with **validation-driven thresholding (maximize TSS)** and optionally **minority-class GAN augmentation (WGAN-GP)**.

---

## 🌍 Overview

Modern infrastructure is vulnerable to severe solar flares (≥M/X), but classification is hard because:
- Severe events are **rare** → extreme class imbalance
- Precursors are **localized** (e.g., polarity inversion lines, sheared structures)
- Relevant cues appear across **multiple atmospheric layers** → multi-wavelength fusion matters

**SunGuard** solves this by keeping things modular:
1. Train **10 independent single-channel classifiers** (one per wavelength)
2. Extract **embeddings** (1280-D each)
3. Fuse embeddings using:
   - **Residual Neural Fusion (1D ResNet)**, or
   - **XGBoost** on concatenated embeddings (strong classical baseline)
4. Pick decision threshold by **maximizing TSS on validation**, then evaluate on test

---

## ✨ Key Features

- **Multi-instrument / multi-wavelength learning**: 8× AIA + 2× HMI (10 channels total)
- **Single-channel → 3-channel adaptation (FFT trick)**:
  - channel-1: original intensity
  - channel-2: FFT magnitude
  - channel-3: FFT phase
- **Pretrained EfficientNetV2-S** per wavelength for strong representation learning
- **Embedding-based fusion**:
  - preserves modularity + allows fair fusion comparisons
- **Imbalance-aware evaluation**:
  - threshold selected by **TSS maximization** (not accuracy)
- **Optional GAN augmentation**:
  - per-wavelength **WGAN-GP** trained only on severe-class images

---

## 🧠 Method (High-Level)

### 1) Data & Channels (SDOBenchmark)
10 synchronized channels per sample:
- AIA: **94, 131, 171, 193, 211, 304, 335, 1700 Å**
- HMI: **Continuum**, **Magnetogram**

Binary labels:
- **Severe** = M/X-class (GOES peak flux ≥ 1e−5 W/m²)
- **Non-severe** = A/B/C-class

### 2) Preprocessing (per wavelength)
- Resize to **128×128**
- Build a **3-channel tensor** via FFT:
  - `[I, log(1+|FFT(I)|), phase(FFT(I))]`
- Normalize using ImageNet mean/std (to match EfficientNet pretraining)

### 3) Single-Wavelength Backbones (10 models)
- Backbone: **EfficientNetV2-S** (ImageNet pretrained)
- Binary head: dropout(0.5) → linear(1)
- Extract embedding from final pooling:
  - **vλ ∈ R¹²⁸⁰** for each wavelength λ

### 4) Fusion (two options)
- **Residual Neural Fusion (1D ResNet)**:
  - stack embeddings → shape **(B, 10, 1280)**
  - 1D conv + residual blocks (1024→512→256→128) → GAP → sigmoid
- **XGBoost Fusion**:
  - concatenate embeddings → **12,800-D**
  - boosted trees with regularization + subsampling

### 5) Thresholding
- choose τ* on validation by:
  - **τ* = argmax TSS(τ) = TPR(τ) − FPR(τ)**
- apply τ* once to test set (no leakage)

---

## 📊 Results

### Baselines (no GAN)
| Fusion Model | TSS (↑) | F1 | Recall | Precision | ROC-AUC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Residual Fusion (ResNet) | **0.412 ± 0.024** | 0.462 ± 0.017 | 0.819 ± 0.038 | 0.323 ± 0.022 | 0.706 ± 0.012 | 0.636 ± 0.043 |
| XGBoost Fusion | **0.500 ± 0.016** | 0.526 ± 0.005 | 0.779 ± 0.055 | 0.399 ± 0.020 | 0.750 ± 0.008 | 0.732 ± 0.022 |

### Effect of GAN augmentation (minority-class only)
Residual Fusion benefits from *moderate* synthetic ratios:

| GAN Ratio | TSS (↑) | F1 | Recall | Precision |
|---:|---:|---:|---:|---:|
| 0%  | 0.412 ± 0.024 | 0.462 ± 0.017 | 0.819 ± 0.038 | 0.323 ± 0.022 |
| 10% | **0.435 ± 0.030** | 0.490 ± 0.013 | 0.725 ± 0.052 | 0.371 ± 0.010 |
| 30% | **0.435 ± 0.017** | 0.492 ± 0.008 | 0.720 ± 0.086 | 0.379 ± 0.037 |
| 40% | **0.436 ± 0.055** | 0.484 ± 0.029 | 0.764 ± 0.061 | 0.355 ± 0.026 |
| 50% | 0.414 ± 0.035 | 0.484 ± 0.013 | 0.673 ± 0.067 | 0.380 ± 0.011 |

XGBoost does **not** improve with synthetic data (slight degradation across ratios).

