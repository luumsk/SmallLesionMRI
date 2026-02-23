# GT_DIR="/Volumes/BACH2TB/Projects/SmallLesionMRI/labelsTs"
# PRED_MASK_DIR="/Volumes/BACH2TB/Projects/SmallLesionMRI/MSLesSeg/nnUNet"
# PRED_PROB_DIR="/Volumes/BACH2TB/Projects/SmallLesionMRI/MSLesSeg/nnUNet"
# OUT_CSV="metrics/multiscale_eval_nnUNet.csv"
# OUT_TXT="metrics/multiscale_eval_nnUNet.txt"

# python multiscale_eval.py \
#   --gt_dir $GT_DIR \
#   --pred_mask_dir $PRED_MASK_DIR \
#   --pred_prob_dir $PRED_PROB_DIR \
#   --out_csv $OUT_CSV \
#   --out_txt $OUT_TXT \
#   --small_voxels_thresh 150 \
# #   --allow_missing_prob


GT_DIR="/Volumes/BACH2TB/Projects/SmallLesionMRI/labelsTs"
BASE_DIR="/Volumes/BACH2TB/Projects/SmallLesionMRI/MSLesSeg"
OUT_DIR="metrics"

MODELS=(
  MIL
  CAT
  CATMIL
  nnUNet
  SegResNet
  UNETR
  SwinUNETR
  UMambaBot
  UMambaEnc
)

for MODEL in "${MODELS[@]}"; do

  echo "Running multiscale evaluation for ${MODEL}..."

  PRED_MASK_DIR="${BASE_DIR}/${MODEL}"
  PRED_PROB_DIR="${BASE_DIR}/${MODEL}"
  OUT_CSV="${OUT_DIR}/multiscale_eval_${MODEL}.csv"
  OUT_TXT="${OUT_DIR}/multiscale_eval_${MODEL}.txt"

  python multiscale_eval.py \
    --gt_dir "${GT_DIR}" \
    --pred_mask_dir "${PRED_MASK_DIR}" \
    --pred_prob_dir "${PRED_PROB_DIR}" \
    --out_csv "${OUT_CSV}" \
    --out_txt "${OUT_TXT}" \
    --small_voxels_thresh 150
    # --allow_missing_prob

done

echo "All models evaluated."