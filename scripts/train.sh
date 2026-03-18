set -e  # stop immediately if any command fails

DATASET_ID="333"
CONFIGURATION="3d_fullres"
DEVICE="cuda"

MODELS=(
    nnUNetTrainerCAT
    nnUNetTrainerCATMIL
    nnUNetTrainer
    nnUNetTrainerMIL
    nnUNetTrainerSegResNet
    nnUNetTrainerUNETR
    nnUNetTrainerSwinUNETR
    nnUNetTrainerUMambaBot
    nnUNetTrainerUMambaEnc
)

for TRAINER in "${MODELS[@]}"
do
    echo "=============================="
    echo "Training model: $TRAINER"
    echo "=============================="

    for FOLD in 0 1 2 3 4
    do
        echo "Fold $FOLD"
        nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD \
            -tr $TRAINER \
            -device $DEVICE
    done

    echo "Finished $TRAINER"
done

echo "All training completed successfully."