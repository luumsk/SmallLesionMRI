"""Test-set inference for one SwinUNETR (DiceCE) fold model on MSLesSeg (no training).

SwinUNETR fold_4 was trained to epoch 150 (checkpoint_final.pth), but its
test-set prediction was never written to
/Volumes/BACH2TB/Tunka/SmallLesionMRI/MSLesSeg/final_checkpoints/SwinUNETR/fold_4.
This script produces it with the stored checkpoint and the nnU-Net fork in
SmallLesionMRI/slsseg (sliding window, step 0.5, Gaussian weighting, mirroring
on axes (0, 1, 2), as nnU-Net's default prediction). The original predictions
were made on CUDA; run fold 0 first and compare with its stored prediction
(check_against).

Usage:
  python -W ignore predict_swinunetr_fold.py <fold> <out_dir> [case ...] [--check <stored_fold_dir>]
"""
import argparse
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "slsseg"))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402
from nnunetv2.inference.export_prediction import export_prediction_from_logits  # noqa: E402
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO  # noqa: E402
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # noqa: E402
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerSwinUNETR import nnUNetTrainerSwinUNETR  # noqa: E402

TUNKA = "/Volumes/BACH2TB/Tunka"
MODEL = f"{TUNKA}/nnUNet_results/Dataset333_MSLesSeg/nnUNetTrainerSwinUNETR__nnUNetPlans__3d_fullres"
RAW = f"{TUNKA}/nnUNet_raw/Dataset333_MSLesSeg"


def case_ids():
    files = os.listdir(os.path.join(RAW, "imagesTs"))
    return sorted({f.rsplit("_", 1)[0] for f in files if f.endswith(".nii.gz")})


def build_predictor(fold):
    ck = torch.load(os.path.join(MODEL, f"fold_{fold}", "checkpoint_final.pth"), map_location="cpu",
                    weights_only=False)
    assert ck["current_epoch"] == 150, ck["current_epoch"]
    plans = json.load(open(os.path.join(MODEL, "plans.json")))
    djson = json.load(open(os.path.join(MODEL, "dataset.json")))
    pm = PlansManager(plans)
    cm = pm.get_configuration(ck["init_args"]["configuration"])
    net = nnUNetTrainerSwinUNETR.build_network_architecture(pm, djson, cm, len(djson["channel_names"]),
                                                            enable_deep_supervision=False)
    net.load_state_dict(ck["network_weights"])
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
                           perform_everything_on_device=False, device=dev, verbose=False, allow_tqdm=False)
    pred.manual_initialization(net, pm, cm, [ck["network_weights"]], djson, "nnUNetTrainerSwinUNETR",
                               ck["inference_allowed_mirroring_axes"])
    pred.network = pred.network.to(dev)
    return pred, djson


def predict_case(pred, djson, case, out_dir):
    # channel keys are "1", "2", "3" -> files _0001.._0003 (same order as the training data)
    chans = sorted(djson["channel_names"], key=int)
    files = [os.path.join(RAW, "imagesTs", f"{case}_{int(c):04d}.nii.gz") for c in chans]
    img, props = SimpleITKIO().read_images(files)
    pp = pred.configuration_manager.preprocessor_class(verbose=False)
    data, _ = pp.run_case_npy(img, None, props, pred.plans_manager, pred.configuration_manager, djson)
    logits = pred.predict_logits_from_preprocessed_data(torch.from_numpy(data).float().contiguous()).cpu()
    export_prediction_from_logits(logits, props, pred.configuration_manager, pred.plans_manager, djson,
                                  os.path.join(out_dir, case), save_probabilities=True)


def dice(a, b):
    s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fold", type=int)
    ap.add_argument("out_dir")
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--check", help="stored prediction folder of the same fold to compare with")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    cases = a.cases or case_ids()
    pred, djson = build_predictor(a.fold)
    for c in cases:
        predict_case(pred, djson, c, a.out_dir)
        msg = f"fold {a.fold} {c} done"
        if a.check:
            new = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(a.out_dir, f"{c}.nii.gz"))) > 0
            old = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(a.check, f"{c}.nii.gz"))) > 0
            msg += (f"  Dice vs stored {dice(new, old):.4f}  voxels new/stored {int(new.sum())}/{int(old.sum())}"
                    f"  differing voxels {int(np.logical_xor(new, old).sum())}")
        print(msg, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
