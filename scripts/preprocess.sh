#!/usr/bin/env bash
# Run from anywhere; paths are relative to the repository root.
cd "$(dirname "$0")/.." && source scripts/config.sh

DATASET_ID="555"
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity