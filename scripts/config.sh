# Paths shared by every script in scripts/. Each one can be overridden in the environment, e.g.
#   SLS_DATA_ROOT=/data/me bash scripts/predict.sh
#
# SLS_DATA_ROOT holds the nnU-Net folders and the predictions:
#   $SLS_DATA_ROOT/nnUNet_raw/  nnUNet_preprocessed/  nnUNet_results/
#   $SLS_DATA_ROOT/SmallLesionMRI/MSLesSeg/final_checkpoints/<MODEL>/fold_<k>/   (predictions)
#   $SLS_DATA_ROOT/SmallLesionMRI/MSLesSeg/features/                            (extracted features)
# The analysis/ Python scripts read the same variable.
: "${SLS_DATA_ROOT:=/media/storage/luu}"
export SLS_DATA_ROOT
export nnUNet_raw="${nnUNet_raw:-$SLS_DATA_ROOT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$SLS_DATA_ROOT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$SLS_DATA_ROOT/nnUNet_results}"
PRED_ROOT="${PRED_ROOT:-$SLS_DATA_ROOT/SmallLesionMRI/MSLesSeg}"

# Our server activates a virtualenv here; activate your own environment before running the scripts.
# source /media/storage/luu/slsseg_venv/bin/activate