#!/usr/bin/env bash
# Run from anywhere; paths are relative to the repository root.
cd "$(dirname "$0")/.." && source scripts/config.sh

PREDICTION_DIR="${PRED_ROOT}/UMambaEnc"
GROUNDTRUTH_DIR="${nnUNet_raw}/Dataset333_MSLesSeg/labelsTs"
OUTPUT_CSV="results/metrics_MSLesSeg_UMambaEnc.csv"

python evaluation/compute_metrics.py --pr $PREDICTION_DIR --gt $GROUNDTRUTH_DIR --out $OUTPUT_CSV