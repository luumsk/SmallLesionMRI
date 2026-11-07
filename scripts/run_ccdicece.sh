#!/bin/bash
# Train nnU-Net with CC-DiceCE (Bouteille et al. 2026) on MSLesSeg (5 folds, 150 epochs),
# predict the test set with each fold model, and run the same multiscale evaluation as the
# other losses. Same splits, plans, epochs and checkpoint as nnUNetTrainer / nnUNetTrainerCATMIL.
#
# Usage (on the server, from SmallLesionMRI/scripts):
#   nohup bash run_ccdicece.sh > ccdicece.log 2>&1 &
#   FOLDS="0 1" bash run_ccdicece.sh      # subset of folds (e.g. two GPUs in parallel)
#   STAGES="predict eval" bash run_ccdicece.sh   # skip training
set -e
cd "$(dirname "$0")"
source setvars.sh

DATASET_ID="333"
CONFIGURATION="3d_fullres"
DEVICE="cuda"
TRAINER="nnUNetTrainerCCDiceCE"
MODEL="CCDiceCE"
CHECKPOINT="checkpoint_final.pth"
FOLDS=(${FOLDS:-0 1 2 3 4})
STAGES="${STAGES:-train predict eval}"

INPUT_FOLDER="/media/storage/luu/nnUNet_raw/Dataset333_MSLesSeg/imagesTs"
GT_DIR="/media/storage/luu/nnUNet_raw/Dataset333_MSLesSeg/labelsTs"
BASE_OUTPUT="/media/storage/luu/SmallLesionMRI/MSLesSeg/final_checkpoints"
OUT_DIR="../metrics/MSLesSeg/latest_checkpoints"

# Fail early if the trainer is not visible to the installed nnunetv2
python -c "from nnunetv2.training.nnUNetTrainer.nnUNetTrainerCCDiceCE import nnUNetTrainerCCDiceCE, _HAS_CUPY; print('Voronoi on', 'GPU (cupy)' if _HAS_CUPY else 'CPU (scipy fallback; pip install cupy-cuda12x for speed)')"

for FOLD in "${FOLDS[@]}"; do
  if [[ " $STAGES " == *" train "* ]]; then
    echo "=== [$(date)] train fold $FOLD"
    RESULT_DIR="$nnUNet_results/Dataset${DATASET_ID}_MSLesSeg/${TRAINER}__nnUNetPlans__${CONFIGURATION}/fold_${FOLD}"
    if [ -f "$RESULT_DIR/checkpoint_final.pth" ]; then
      echo "fold $FOLD already trained, skipping"
    elif [ -f "$RESULT_DIR/checkpoint_latest.pth" ]; then
      nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD -tr $TRAINER -device $DEVICE --c
    else
      nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD -tr $TRAINER -device $DEVICE
    fi
  fi

  if [[ " $STAGES " == *" predict "* ]]; then
    echo "=== [$(date)] predict fold $FOLD"
    OUTPUT_FOLDER="${BASE_OUTPUT}/${MODEL}/fold_${FOLD}"
    mkdir -p "$OUTPUT_FOLDER"
    nnUNetv2_predict -i "$INPUT_FOLDER" -o "$OUTPUT_FOLDER" \
      -d "$DATASET_ID" -c "$CONFIGURATION" -f "$FOLD" -tr "$TRAINER" \
      -device "$DEVICE" -chk "$CHECKPOINT" --save_probabilities
  fi

  if [[ " $STAGES " == *" eval "* ]]; then
    echo "=== [$(date)] evaluate fold $FOLD"
    mkdir -p "$OUT_DIR"
    python ../multiscale_eval.py \
      --gt_dir "$GT_DIR" \
      --pred_mask_dir "${BASE_OUTPUT}/${MODEL}/fold_${FOLD}" \
      --pred_prob_dir "${BASE_OUTPUT}/${MODEL}/fold_${FOLD}" \
      --out_csv "${OUT_DIR}/multiscale_eval_${MODEL}_fold${FOLD}.csv" \
      --out_json "${OUT_DIR}/multiscale_eval_${MODEL}_fold${FOLD}.json" \
      --small_voxels_thresh 150
  fi
done

echo "=== [$(date)] done: $MODEL folds ${FOLDS[*]}"
