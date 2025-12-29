PREDICTION_DIR="/media/storage/luu/SmallLesionMRI/MSLesSeg/UMambaEnc"
GROUNDTRUTH_DIR="/media/storage/luu/nnUNet_raw/Dataset333_MSLesSeg/labelsTs"
OUTPUT_CSV="./metrics/metrics_MSLesSeg_UMambaEnc.csv"

python compute_metrics.py --pr $PREDICTION_DIR --gt $GROUNDTRUTH_DIR --out $OUTPUT_CSV