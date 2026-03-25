python analyze_features.py \
  --catmil_feature_root /media/storage/luu/SmallLesionMRI/MSLesSeg/features/CATMIL/fold_0/features_decoder_seg_layers_4_input/ \
  --nnunet_feature_root /media/storage/luu/SmallLesionMRI/MSLesSeg/features/nnUNet/fold_0/features_decoder_seg_layers_4_input/ \
  --gt_root /media/storage/luu/nnUNet_raw/Dataset333_MSLesSeg/labelsTs \
  --output_root /media/storage/luu/SmallLesionMRI/MSLesSeg/analysis_out \
  --layer_dir decoder_seg_layers_4 \
  --catmil_pred_root /media/storage/luu/SmallLesionMRI/MSLesSeg/final_checkpoints/CATMIL/fold_0 \
  --nnunet_pred_root /media/storage/luu/SmallLesionMRI/MSLesSeg/final_checkpoints/nnUNet/fold_0