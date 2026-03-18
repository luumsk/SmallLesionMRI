GT_DIR="/media/storage/luu/nnUNet_raw/Dataset333_MSLesSeg/labelsTs"
BASE_DIR="/media/storage/luu/SmallLesionMRI/MSLesSeg/final_checkpoints"
OUT_DIR="metrics/MSLesSeg/latest_checkpoints"

FOLDS=(0 1 2 3 4)

MODELS=(
  # CAT
  # MIL
  # CATMIL
  # nnUNet
  # SegResNet
  # UNETR
  SwinUNETR
  # UMambaBot
  # UMambaEnc
)

mkdir -p "$OUT_DIR"

for MODEL in "${MODELS[@]}"; do

  echo "Running multiscale evaluation for ${MODEL}..."

  for FOLD in "${FOLDS[@]}"; do

    echo "  Fold ${FOLD}"

    PRED_MASK_DIR="${BASE_DIR}/${MODEL}/fold_${FOLD}"
    PRED_PROB_DIR="${BASE_DIR}/${MODEL}/fold_${FOLD}"

    OUT_CSV="${OUT_DIR}/multiscale_eval_${MODEL}_fold${FOLD}.csv"
    OUT_TXT="${OUT_DIR}/multiscale_eval_${MODEL}_fold${FOLD}.txt"

    python multiscale_eval.py \
      --gt_dir "${GT_DIR}" \
      --pred_mask_dir "${PRED_MASK_DIR}" \
      --pred_prob_dir "${PRED_PROB_DIR}" \
      --out_csv "${OUT_CSV}" \
      --out_txt "${OUT_TXT}" \
      --small_voxels_thresh 150
      # --allow_missing_prob

  done

done

echo "All models evaluated."