#!/bin/bash
# Train nnU-Net with CC-DiceCE (Bouteille et al. 2026) on MSLesSeg or 3D-MR-MS (5 folds, 150 epochs),
# predict the test set with each fold model, and run the same multiscale evaluation as the
# other losses. Same splits, plans, epochs and checkpoint as nnUNetTrainer / nnUNetTrainerCATMIL.
#
# Usage (on the server, from SmallLesionMRI/scripts):
#   PYTHONUNBUFFERED=1 bash run_ccdicece.sh 2>&1 | tee ccdicece.log                  # MSLesSeg (default)
#   DATASET=LjubljanaMS PYTHONUNBUFFERED=1 bash run_ccdicece.sh 2>&1 | tee ccdicece_ljubljana.log
#   FOLDS="0 1" bash run_ccdicece.sh              # subset of folds
#   STAGES="predict eval" bash run_ccdicece.sh   # skip training
set -e
cd "$(dirname "$0")"
source setvars.sh

DATASET="${DATASET:-MSLesSeg}"
case "$DATASET" in
  MSLesSeg)    DATASET_ID="333"; DATASET_NAME="Dataset333_MSLesSeg";    METRICS_MODES="${METRICS_MODES:-main}" ;;
  LjubljanaMS) DATASET_ID="666"; DATASET_NAME="Dataset666_LjubljanaMS"; METRICS_MODES="${METRICS_MODES:-main all}" ;;
  *) echo "unknown DATASET=$DATASET (use MSLesSeg or LjubljanaMS)"; exit 1 ;;
esac
CONFIGURATION="3d_fullres"
DEVICE="cuda"
TRAINER="nnUNetTrainerCCDiceCE"
MODEL="CCDiceCE"
CHECKPOINT="checkpoint_final.pth"
FOLDS=(${FOLDS:-0 1 2 3 4})
STAGES="${STAGES:-train predict eval}"

INPUT_FOLDER="$nnUNet_raw/${DATASET_NAME}/imagesTs"
GT_DIR="$nnUNet_raw/${DATASET_NAME}/labelsTs"
BASE_OUTPUT="/media/storage/luu/SmallLesionMRI/${DATASET}/final_checkpoints"
OUT_DIR="../metrics/${DATASET}/latest_checkpoints"
echo "dataset: $DATASET_NAME | test images: $INPUT_FOLDER | predictions: $BASE_OUTPUT/$MODEL"
for d in "$nnUNet_preprocessed/$DATASET_NAME" "$INPUT_FOLDER" "$GT_DIR"; do
  [ -e "$d" ] || { echo "!! missing: $d"; exit 1; }
done
[ -f "$nnUNet_preprocessed/$DATASET_NAME/splits_final.json" ] || { echo "!! missing splits_final.json (needed for the same folds as the other losses)"; exit 1; }

# Fail early if the trainer is not visible to the installed nnunetv2
python - <<'PY'
import importlib, torch
m = importlib.import_module('nnunetv2.training.nnUNetTrainer.nnUNetTrainerCCDiceCE')
_voronoi = m._voronoi
if m._HAS_CUPY and torch.cuda.is_available():
    y = torch.zeros(32, 32, 32, dtype=torch.bool, device='cuda'); y[4:8, 4:8, 4:8] = 1; y[20:22, 20:22, 20:22] = 1
    v = _voronoi(y, (1.0, 1.0, 1.0))  # self-test of the GPU path; falls back to CPU if it fails
    assert int(v.max()) == 2 and v.min() > 0
print('Voronoi on', 'GPU (cupy)' if m._HAS_CUPY else 'CPU (scipy; pip install cupy-cuda11x / cupy-cuda12x for speed)')
PY

for FOLD in "${FOLDS[@]}"; do
  if [[ " $STAGES " == *" train "* ]]; then
    echo "=== [$(date)] train fold $FOLD"
    RESULT_DIR="$nnUNet_results/${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIGURATION}/fold_${FOLD}"
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
    for MODE in $METRICS_MODES; do   # output files get the suffix _main / _all
      python ../multiscale_eval.py \
        --gt_dir "$GT_DIR" \
        --pred_mask_dir "${BASE_OUTPUT}/${MODEL}/fold_${FOLD}" \
        --pred_prob_dir "${BASE_OUTPUT}/${MODEL}/fold_${FOLD}" \
        --out_csv "${OUT_DIR}/multiscale_eval_${MODEL}_fold${FOLD}.csv" \
        --out_json "${OUT_DIR}/multiscale_eval_${MODEL}_fold${FOLD}.json" \
        --small_voxels_thresh 150 \
        --metrics_mode "$MODE"
    done
  fi
done

echo "=== [$(date)] done: $MODEL folds ${FOLDS[*]}"
