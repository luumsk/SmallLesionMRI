#!/usr/bin/env bash
# Run from anywhere; paths are relative to the repository root.
cd "$(dirname "$0")/.." && source scripts/config.sh

python evaluation/get_all_metrics.py \
  -i "results/MSLesSeg/main" \
  -o "all_metrics.csv"