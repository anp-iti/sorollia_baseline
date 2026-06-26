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
sorollia
	├── Ground-Truth
	|		├── CV
	|		├── Non-CV
	|		└── fold_assignments.csv
	└── audios
		├── 00001.flac
		.....
		└── 07396.flac
```

## Run 5-fold benchmark

To reproduce the full benchmark, use the provided scripts. They automatically generate the train/test CSV files, pack waveforms into HDF5, create indexes, and run training or fine-tuning.

Both evaluation protocols are supported through `--fold-group`:

- `cv`: run only CV folds
- `non_cv`: run only Non-CV folds
- `both`: run both protocols, default option

### From scratch

```bash
bash scripts/run_benchmark_scratch.sh --fold-group <cv|non_cv|both>
```

### Fine-tune CNN14

```bash
bash scripts/run_benchmark_finetune.sh --fold-group <cv|non_cv|both>
```

### Generate train/test CSVs only

If you only want to generate the train/test CSV files for all folds, run:

```bash
bash scripts/generate_all_folds_csv.sh
```

This creates the train.csv and test.csv files under:

```text
sorollia/Ground-Truth/generated/
├── CV/
└── Non-CV/
```

## Step by Step

Supposing using CV fold 0 for testing

### Create Train/Test csvs

The dataset provides one CSV file per fold. Before preprocessing, explicit train.csv and test.csv files must be generated.

For CV fold 0:

```bash
python src/utils/csv_from_folds.py \
    -d sorollia/Ground-Truth \
    --fold-type cv \
    -t 0 \
    --train-out train.csv \
    --test-out test.csv \
    -o sorollia/Ground-Truth/generated/CV/fold_0
```

This creates:

```text
sorollia/Ground-Truth/generated/CV/fold_0/
├── train.csv
└── test.csv
```

### Data Preprocessing

To prepare the :

```bash
python src/utils/dataset.py \
	pack_waveforms_to_hdf5 \
	--csv_path=sorollia/Ground-Truth/generated/CV/fold_0/train.csv \    --logs_path=logs/CV/fold_0 \
	--audio_dir=sorollia/audios \
	--waveforms_hdf5_path=hdf5s/CV/fold_0/waveforms/train.h5 \
	--csv_label=labels_sorollia.csv \
	--fsamp=32000

python src/utils/dataset.py \
	pack_waveforms_to_hdf5 \
	--csv_path=sorollia/Ground-Truth/generated/CV/fold_0/test.csv \    --logs_path=logs/CV/fold_0 \
	--audio_dir=sorollia/audios \
	--waveforms_hdf5_path=hdf5s/CV/fold_0/waveforms/test.h5 \
	--csv_label=labels_sorollia.csv \
	--fsamp=32000
```

```bash
python src/utils/create_indexes.py \
	create_indexes \
	--waveforms_hdf5_path=hdf5s/CV/fold_0/waveforms/train.h5 \
	--indexes_hdf5_path=hdf5s/CV/fold_0/indexes/train.h5

python src/utils/create_indexes.py \
	create_indexes \
	--waveforms_hdf5_path=hdf5s/CV/fold_0/waveforms/test.h5 \
	--indexes_hdf5_path=hdf5s/CV/fold_0/indexes/test.h5
```

### Training

```bash
python src/main.py \
	--workspace=outputs/CV/fold_0 \ 
	--train_data=hdf5s/CV/fold_0/indexes/train.h5 \
	--test_data=hdf5s/CV/fold_0/indexes/test.h5 \
	--csv_label=labels_sorollia.csv
```

```bash
python src/finetune.py \
	--workspace=outputs_finetune/CV/fold_0 \
	--train_data=hdf5s/CV/fold_0/indexes/train.h5 \
	--test_data=hdf5s/CV/fold_0/indexes/test.h5 \
	--csv_label=labels_sorollia.csv
```

# Citation

If using this dataset, please cite:

1. Naranjo-Alcazar, J., Grau-Haro, J., Ribes-Serrano, R., & Zuccarello, P. (2025, March). A Data-Centric Framework for Machine Listening Projects: Addressing Large-Scale Data Acquisition and Labeling through Active Learning. In Future of Information and Communication Conference (pp. 647-659). Cham: Springer Nature Switzerland.

```text
@inproceedings{naranjo2025data,
  title={A Data-Centric Framework for Machine Listening Projects: Addressing Large-Scale Data Acquisition and Labeling through Active Learning},
  author={Naranjo-Alcazar, Javier and Grau-Haro, Jordi and Ribes-Serrano, Ruben and Zuccarello, Pedro},
  booktitle={Future of Information and Communication Conference},
  pages={647--659},
  year={2025},
  organization={Springer}
}
```

2. Soroll-IA paper to be added