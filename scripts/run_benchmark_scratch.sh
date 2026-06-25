#!/bin/bash

set -e

# Move to repository root, regardless of where the script is launched from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# ============================================================
# Configuration
# ============================================================

DATASET_DIR="sorollia"
GROUND_TRUTH_DIR="${DATASET_DIR}/Ground-Truth"
GENERATED_FOLDS_DIR="${GROUND_TRUTH_DIR}/generated"

AUDIO_DIR="${DATASET_DIR}/audios"
CSV_LABEL="labels_sorollia.csv"

WORKSPACE_BASE="outputs"
HDF5_BASE="hdf5s"

FSAMP=32000

CSV_FROM_FOLDS_SCRIPT="src/utils/csv_from_folds.py"
DATASET_SCRIPT="src/utils/dataset.py"
INDEX_SCRIPT="src/utils/create_indexes.py"
TRAIN_SCRIPT="src/pytorch/main.py"

# Extra training arguments.
# Add here the arguments required by your src/pytorch/main.py.
TRAIN_ARGS=""

# Example:
# TRAIN_ARGS="--model_type=Cnn14 --batch_size=32 --learning_rate=1e-3 --cuda"


# ============================================================
# Arguments
# ============================================================

FOLD_GROUP_ARG="both"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --fold-group)
            FOLD_GROUP_ARG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage:"
            echo "  bash scripts/run_benchmark_scratch.sh --fold-group [cv|non_cv|both]"
            echo ""
            echo "Examples:"
            echo "  bash scripts/run_benchmark_scratch.sh --fold-group cv"
            echo "  bash scripts/run_benchmark_scratch.sh --fold-group non_cv"
            echo "  bash scripts/run_benchmark_scratch.sh --fold-group both"
            echo ""
            echo "If --fold-group is not provided, both CV and Non-CV will be processed."
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Use --help to see usage."
            exit 1
            ;;
    esac
done

case "${FOLD_GROUP_ARG}" in
    cv)
        FOLD_GROUPS=("CV")
        ;;
    non_cv)
        FOLD_GROUPS=("Non-CV")
        ;;
    both)
        FOLD_GROUPS=("CV" "Non-CV")
        ;;
    *)
        echo "ERROR: Invalid --fold-group value: ${FOLD_GROUP_ARG}"
        echo "Valid values are: cv, non_cv, both"
        exit 1
        ;;
esac


# ============================================================
# Checks
# ============================================================

if [ ! -d "${DATASET_DIR}" ]; then
    echo "ERROR: Dataset directory not found: ${DATASET_DIR}"
    exit 1
fi

if [ ! -d "${GROUND_TRUTH_DIR}" ]; then
    echo "ERROR: Ground-truth directory not found: ${GROUND_TRUTH_DIR}"
    exit 1
fi

if [ ! -d "${AUDIO_DIR}" ]; then
    echo "ERROR: Audio directory not found: ${AUDIO_DIR}"
    exit 1
fi

if [ ! -f "${CSV_LABEL}" ]; then
    echo "ERROR: Label CSV not found: ${CSV_LABEL}"
    exit 1
fi

if [ ! -f "${CSV_FROM_FOLDS_SCRIPT}" ]; then
    echo "ERROR: CSV generation script not found: ${CSV_FROM_FOLDS_SCRIPT}"
    exit 1
fi

if [ ! -f "${DATASET_SCRIPT}" ]; then
    echo "ERROR: Dataset script not found: ${DATASET_SCRIPT}"
    exit 1
fi

if [ ! -f "${INDEX_SCRIPT}" ]; then
    echo "ERROR: Index script not found: ${INDEX_SCRIPT}"
    exit 1
fi

if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo "ERROR: Training script not found: ${TRAIN_SCRIPT}"
    exit 1
fi


# ============================================================
# Pipeline
# ============================================================

echo ""
echo "============================================================"
echo "Running benchmark from scratch"
echo "Fold group argument: ${FOLD_GROUP_ARG}"
echo "Dataset directory: ${DATASET_DIR}"
echo "Audio directory: ${AUDIO_DIR}"
echo "Generated folds directory: ${GENERATED_FOLDS_DIR}"
echo "HDF5 base directory: ${HDF5_BASE}"
echo "Workspace base directory: ${WORKSPACE_BASE}"
echo "============================================================"

for FOLD_GROUP in "${FOLD_GROUPS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Processing fold group: ${FOLD_GROUP}"
    echo "============================================================"

    if [ "${FOLD_GROUP}" = "CV" ]; then
        FOLD_TYPE_ARG="cv"
    else
        FOLD_TYPE_ARG="non_cv"
    fi

    for TEST_FOLD in 0 1 2 3 4; do
        echo ""
        echo "------------------------------------------------------------"
        echo "Processing ${FOLD_GROUP} fold_${TEST_FOLD}"
        echo "------------------------------------------------------------"

        CSV_DIR="${GENERATED_FOLDS_DIR}/${FOLD_GROUP}/fold_${TEST_FOLD}"
        TRAIN_CSV="${CSV_DIR}/train.csv"
        TEST_CSV="${CSV_DIR}/test.csv"

        WAVEFORMS_DIR="${HDF5_BASE}/${FOLD_GROUP}/fold_${TEST_FOLD}/waveforms"
        INDEXES_DIR="${HDF5_BASE}/${FOLD_GROUP}/fold_${TEST_FOLD}/indexes"

        TRAIN_WAVEFORMS_HDF5="${WAVEFORMS_DIR}/train.h5"
        TEST_WAVEFORMS_HDF5="${WAVEFORMS_DIR}/test.h5"

        TRAIN_INDEXES_HDF5="${INDEXES_DIR}/train.h5"
        TEST_INDEXES_HDF5="${INDEXES_DIR}/test.h5"

        WORKSPACE="${WORKSPACE_BASE}/${FOLD_GROUP}/fold_${TEST_FOLD}"

        mkdir -p "${CSV_DIR}"
        mkdir -p "${WAVEFORMS_DIR}"
        mkdir -p "${INDEXES_DIR}"
        mkdir -p "${WORKSPACE}"

        echo ""
        echo "[1/6] Generating train/test CSV files"
        python "${CSV_FROM_FOLDS_SCRIPT}" \
            -d "${GROUND_TRUTH_DIR}" \
            --fold-type "${FOLD_TYPE_ARG}" \
            -t "${TEST_FOLD}" \
            --train-out train.csv \
            --test-out test.csv \
            -o "${CSV_DIR}"

        echo ""
        echo "[2/6] Packing train waveforms"
        python "${DATASET_SCRIPT}" \
            pack_waveforms_to_hdf5 \
            --csv_path="${TRAIN_CSV}" \
            --audio_dir="${AUDIO_DIR}" \
            --waveforms_hdf5_path="${TRAIN_WAVEFORMS_HDF5}" \
            --csv_label="${CSV_LABEL}" \
            --fsamp="${FSAMP}"

        echo ""
        echo "[3/6] Packing test waveforms"
        python "${DATASET_SCRIPT}" \
            pack_waveforms_to_hdf5 \
            --csv_path="${TEST_CSV}" \
            --audio_dir="${AUDIO_DIR}" \
            --waveforms_hdf5_path="${TEST_WAVEFORMS_HDF5}" \
            --csv_label="${CSV_LABEL}" \
            --fsamp="${FSAMP}"

        echo ""
        echo "[4/6] Creating train indexes"
        python "${INDEX_SCRIPT}" \
            create_indexes \
            --waveforms_hdf5_path="${TRAIN_WAVEFORMS_HDF5}" \
            --indexes_hdf5_path="${TRAIN_INDEXES_HDF5}"

        echo ""
        echo "[5/6] Creating test indexes"
        python "${INDEX_SCRIPT}" \
            create_indexes \
            --waveforms_hdf5_path="${TEST_WAVEFORMS_HDF5}" \
            --indexes_hdf5_path="${TEST_INDEXES_HDF5}"

        echo ""
        echo "[6/6] Training from scratch"
        python "${TRAIN_SCRIPT}" \
            --workspace="${WORKSPACE}" \
            --train_data="${TRAIN_INDEXES_HDF5}" \
            --test_data="${TEST_INDEXES_HDF5}" \
            --csv_label="${CSV_LABEL}" \
            ${TRAIN_ARGS}

        echo ""
        echo "[OK] Finished ${FOLD_GROUP} fold_${TEST_FOLD}"
    done
done

echo ""
echo "============================================================"
echo "Benchmark from scratch completed successfully."
echo "============================================================"