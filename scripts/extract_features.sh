INPUT_FOLDER="/media/storage/luu/nnUNet_raw/Dataset333_MSLesSeg/imagesTs"
BASE_OUTPUT="/media/storage/luu/SmallLesionMRI/MSLesSeg/features"

DATASET_ID="333"
CONFIGURATION="3d_fullres"
DEVICE="cuda"
CHECKPOINT="checkpoint_final.pth"

FOLDS=(0)

MODELS=(
  # "CAT"
  # "MIL"
  CATMIL
  # "nnUNet"
  # "SegResNet"
  # "UNETR"
  # "SwinUNETR"
  # "UMambaBot"
  # "UMambaEnc"
  # Tversky
  # FocalTversky

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

    echo "Extracting features"
    echo "Model: $MODEL"
    echo "Trainer: $TRAINER"
    echo "Fold: $FOLD"
    echo "Output: $OUTPUT_FOLDER"
    echo "----------------------------------"

    python -m nnunetv2.inference.extract_features \
      -i "$INPUT_FOLDER" \
      -o "$OUTPUT_FOLDER" \
      -d "$DATASET_ID" \
      -c "$CONFIGURATION" \
      -f "$FOLD" \
      -tr "$TRAINER" \
      -device "$DEVICE" \
      -chk "$CHECKPOINT" \
      --layers decoder.seg_layers.4 \
      --capture input \
      --feature_output_dir "$OUTPUT_FOLDER/features_decoder_seg_layers_4_input"

    # python -m nnunetv2.inference.extract_features \
    #   -i "$INPUT_FOLDER" \
    #   -o "$OUTPUT_FOLDER" \
    #   -d "$DATASET_ID" \
    #   -c "$CONFIGURATION" \
    #   -f "$FOLD" \
    #   -tr "$TRAINER" \
    #   -device "$DEVICE" \
    #   -chk "$CHECKPOINT" \
    #   --layers encoder.stages.5.0 \
    #   --capture output \
    #   --feature_output_dir "$OUTPUT_FOLDER/features_encoder_stages_5_0_output"

  done

done

