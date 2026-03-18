INPUT_FOLDER="/media/storage/luu/nnUNet_raw/Dataset333_MSLesSeg/imagesTs"
BASE_OUTPUT="/media/storage/luu/SmallLesionMRI/MSLesSeg/final_checkpoints"

DATASET_ID="333"
CONFIGURATION="3d_fullres"
DEVICE="cuda"
CHECKPOINT="checkpoint_final.pth"

FOLDS=(0 1 2 3 4)

MODELS=(
  # "CAT"
  # "MIL"
  # "CATMIL"
  # "nnUNet"
  # "SegResNet"
  # "UNETR"
  "SwinUNETR"
  # "UMambaBot"
  # "UMambaEnc"
)

for MODEL in "${MODELS[@]}"; do

  if [ "$MODEL" = "nnUNet" ]; then
    TRAINER="nnUNetTrainer"
  else
    TRAINER="nnUNetTrainer${MODEL}"
  fi

  for FOLD in "${FOLDS[@]}"; do

    OUTPUT_FOLDER="${BASE_OUTPUT}/${MODEL}/fold_${FOLD}"
    mkdir -p "$OUTPUT_FOLDER"

    echo "Running prediction"
    echo "Model: $MODEL"
    echo "Trainer: $TRAINER"
    echo "Fold: $FOLD"
    echo "Output: $OUTPUT_FOLDER"
    echo "----------------------------------"

    nnUNetv2_predict \
      -i "$INPUT_FOLDER" \
      -o "$OUTPUT_FOLDER" \
      -d "$DATASET_ID" \
      -c "$CONFIGURATION" \
      -f "$FOLD" \
      -tr "$TRAINER" \
      -device "$DEVICE" \
      -chk "$CHECKPOINT" \
      --save_probabilities

  done

done