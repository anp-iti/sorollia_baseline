# Soroll-IA: Industrial Port Audio Tagging Dataset & Benchmark

## Overview

Soroll-IA is a weakly labeled environmental audio dataset recorded in a real-world industrial port environment. It is designed for multi-label audio tagging under realistic acoustic conditions and supports benchmarking of deep learning models under both cross-validation (CV) and non-cross-validation (Non-CV) evaluation protocols.

The repository provides:
- Dataset processing pipeline
- Training and evaluation scripts
- Benchmark models (CNN14, MobileNetV2)
- Fine-tuning configurations
- Cross-validation evaluation framework

---

## Data

The dataset consists of approximately **22 hours of audio**, segmented into short clips of fixed duration. It includes **26 sound event classes** related to industrial port activity.

### Key properties:
- Multi-label annotations
- Weak labeling (clip-level, no temporal boundaries)
- Two recording nodes deployed in different acoustic zones
- Year-long acquisition period

### Evaluation splits:
- Cross-Validation (CV): fold-based evaluation (5 folds)
- Non-CV: full dataset evaluation without splitting

---

## Data Preparation


## Run 5-fold benchmark

### From scratch

### Fine-tune CNN14


## Step by Step

### Data Preprocessing

To prepare the :

```bash
python src/utils/dataset.py \
```

```bash
python src/utils/create_indexes.py \
```

### Training

