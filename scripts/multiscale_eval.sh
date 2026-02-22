GT_DIR="/Volumes/BACH2TB/Projects/SmallLesionMRI/labelsTs"
PRED_MASK_DIR="/Volumes/BACH2TB/Projects/SmallLesionMRI/MSLesSeg/UMambaEnc"
PRED_PROB_DIR="/Volumes/BACH2TB/Projects/SmallLesionMRI/MSLesSeg/UMambaEnc"
OUT_CSV="metrics/multiscale_eval_UMambaEnc.csv"

python multiscale_eval.py \
  --gt_dir $GT_DIR \
  --pred_mask_dir $PRED_MASK_DIR \
  --pred_prob_dir $PRED_PROB_DIR \
  --out_csv $OUT_CSV \
#   --allow_missing_prob