#!/usr/bin/env bash
# Run from anywhere; paths are relative to the repository root.
cd "$(dirname "$0")/.." && source scripts/config.sh

python analysis/select_test_cases_for_visualization.py \
  --metrics_dir results/MSLesSeg/main \
  --target_model CATMIL \
  --n_improve 4 \
  --n_failure 3 \
  --n_typical 3 \
  --out_csv qualitative_case_selection.csv