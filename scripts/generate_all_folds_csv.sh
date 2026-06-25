#!/bin/bash

set -e

# Base directory containing CV/ and Non-CV/
GROUND_TRUTH_DIR="sorollia/Ground-Truth"

# Script that generates train.csv and test.csv
CSV_SCRIPT="src/utils/csv_from_folds.py"

# Output base directory
OUTPUT_BASE_DIR="${GROUND_TRUTH_DIR}/generated"

echo "==================================="
echo "Generating train/test CSV files"
echo "Ground-truth directory: ${GROUND_TRUTH_DIR}"
echo "CSV script: ${CSV_SCRIPT}"
echo "Output directory: ${OUTPUT_BASE_DIR}"
echo "==================================="

for FOLD_TYPE in cv non_cv; do
    if [ "$FOLD_TYPE" = "cv" ]; then
        OUTPUT_GROUP="CV"
    else
        OUTPUT_GROUP="Non-CV"
    fi

    echo ""
    echo "==================================="
    echo "Processing ${OUTPUT_GROUP}"
    echo "==================================="

    for TEST_FOLD in 0 1 2 3 4; do
        OUTPUT_DIR="${OUTPUT_BASE_DIR}/${OUTPUT_GROUP}/fold_${TEST_FOLD}"

        echo ""
        echo "Generating ${OUTPUT_GROUP} fold ${TEST_FOLD}"
        echo "Output: ${OUTPUT_DIR}"

        python src/utils/csv_from_folds.py \
            -d "${GROUND_TRUTH_DIR}" \
            --fold-type "${FOLD_TYPE}" \
            -t "${TEST_FOLD}" \
            --train-out train.csv \
            --test-out test.csv \
            -o "${OUTPUT_DIR}"
    done
done

echo ""
echo "==================================="
echo "All folds generated successfully."
echo "==================================="