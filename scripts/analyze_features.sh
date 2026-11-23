#!/usr/bin/env bash
# Run from anywhere; paths are relative to the repository root.
cd "$(dirname "$0")/.." && source scripts/config.sh

python analysis/analyze_features.py \
  --catmil_feature_root ${PRED_ROOT}/features/CATMIL/fold_0/features_decoder_seg_layers_4_input/ \
  --nnunet_feature_root ${PRED_ROOT}/features/nnUNet/fold_0/features_decoder_seg_layers_4_input/ \
  --gt_root ${nnUNet_raw}/Dataset333_MSLesSeg/labelsTs \
  --output_root ${PRED_ROOT}/analysis_out \
  --layer_dir decoder_seg_layers_4 \
  --catmil_pred_root ${PRED_ROOT}/final_checkpoints/CATMIL/fold_0 \
  --nnunet_pred_root ${PRED_ROOT}/final_checkpoints/nnUNet/fold_0