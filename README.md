# :anchor: :ship: Soroll-IA: Industrial Port Audio Tagging Dataset & Benchmark

## :sunrise: Overview

Soroll-IA is a weakly labeled environmental audio dataset recorded in a real-world industrial port environment. It is designed for multi-label audio tagging under realistic acoustic conditions and supports benchmarking of deep learning models under both cross-validation (CV) and non-cross-validation (Non-CV) evaluation protocols for annotations.

The repository provides:
- Dataset processing pipeline
- Training and evaluation scripts
- Benchmark models (CNN14, MobileNetV2)
- Fine-tuning configurations
- Cross-validation evaluation framework

---

## :dart: Data

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

The Dataset is provided under [CC BY-NC 4.0 Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)

Once the dataset is download, unzip the data files and stored them on the same path. Your data structure should look like this:

```text
├── Ground-Truth
|	├── CV
|	├── Non-CV
|	└── fold_assignments.csv
└── Audios
	├── 00001.flac
	.....
	└── 07396.flac
```

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

Supposing using fold 0 for testing

### Data Preprocessing

To prepare the :

```bash
python src/utils/dataset.py \
```

```bash
python src/utils/create_indexes.py \
	--waveforms_hdf5_path=waveforms/train.h5 \
	--indexes_hdf5_path=indexes/train.h5

python src/utils/create_indexes.py \
	--waveforms_hdf5_path=waveforms/test.h5 \
	--indexes_hdf5_path=indexes/test.h5
```


### Training

```bash
python src/main.py \
	--workspace=outputs \
	--train_data=indexes/train.h5 \
	--test_data=indexes/test.h5 \
	--csv_label=labels_sorollia.csv \
```

```bash
python src/finetune.py \
	--workspace=outputs \
	--train_data=indexes/train.h5 \
	--test_data=indexes/test.h5 \
	--csv_label=labels_sorollia.csv \
```

