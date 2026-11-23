#!/usr/bin/env bash
# Run from anywhere; paths are relative to the repository root.
cd "$(dirname "$0")/.." && source scripts/config.sh

GT_DIR="${nnUNet_raw}/Dataset333_MSLesSeg/labelsTs"
BASE_DIR="${PRED_ROOT}/final_checkpoints"
OUT_DIR="results/MSLesSeg/latest_checkpoints"

FOLDS=(0 1 2 3 4)

MODELS=(
  CAT
  MIL
  CATMIL
  nnUNet
  SegResNet
  UNETR
  SwinUNETR
  UMambaBot
  UMambaEnc
  Tversky
  FocalTversky
)

mkdir -p "${OUT_DIR}"

for MODEL in "${MODELS[@]}"; do
  echo "Running multiscale evaluation for ${MODEL}..."

  for FOLD in "${FOLDS[@]}"; do
    echo "  Fold ${FOLD}"

    PRED_MASK_DIR="${BASE_DIR}/${MODEL}/fold_${FOLD}"
    PRED_PROB_DIR="${BASE_DIR}/${MODEL}/fold_${FOLD}"

    OUT_CSV="${OUT_DIR}/multiscale_eval_${MODEL}_fold${FOLD}.csv"
    OUT_JSON="${OUT_DIR}/multiscale_eval_${MODEL}_fold${FOLD}.json"

    python evaluation/multiscale_eval.py \
      --gt_dir "${GT_DIR}" \
      --pred_mask_dir "${PRED_MASK_DIR}" \
      --pred_prob_dir "${PRED_PROB_DIR}" \
      --out_csv "${OUT_CSV}" \
      --out_json "${OUT_JSON}" \
      --small_voxels_thresh 150
      # --allow_missing_prob
  done
done

echo "All models evaluated."