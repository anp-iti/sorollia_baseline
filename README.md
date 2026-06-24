# Soroll-IA: Industrial Port Audio Tagging Dataset & Benchmark

## Overview

Soroll-IA is a weakly labeled environmental audio dataset recorded in a real-world industrial port environment. It is designed for multi-label audio tagging under realistic acoustic conditions and supports benchmarking of deep learning models under both cross-validation (CV) and non-cross-validation (Non-CV) evaluation protocols for annotations.

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
- Cross-Validation (CV): at least two-thirds of the annotators must agree on a specific label for an audio clip
- Non-CV: all annotations are assigned to an audio clip

---

## Data Preparation

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/itiresearch/soroll-ia-weakly-labeled-audio-port-monitoring/)

## Run 5-fold benchmark

To reproduce paper benchmark follow with one step, follow the next sections

### From scratch

```bash
sh run_benchmark_scratch.sh
```

### Fine-tune CNN14

```bash
sh run_benchmark_finetune.sh
```

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

```
python src/main.py \
```

```
python src/finetune.py \
```
