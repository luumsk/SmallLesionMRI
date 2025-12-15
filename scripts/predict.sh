INPUT_FOLDER="/media/storage/luu/nnUNet_raw/Dataset333_MSLesSeg/imagesTs"
OUTPUT_FOLDER="/media/storage/luu/SmallLesionMRI/MSLesSeg/SegResNet"
DATASET_ID="333"
CONFIGURATION="3d_fullres"
TRAINER="nnUNetTrainerSegResNet"
DEVICE="cuda"
FOLD="all"
CHECKPOINT="checkpoint_latest.pth"


nnUNetv2_predict -i $INPUT_FOLDER -o $OUTPUT_FOLDER -d $DATASET_ID -c $CONFIGURATION -f $FOLD -tr $TRAINER -device $DEVICE -chk $CHECKPOINT