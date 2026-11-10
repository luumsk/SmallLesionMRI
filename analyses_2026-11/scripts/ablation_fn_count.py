"""Missed lesions per scan for the CAT / MIL ablation on MSLesSeg.

The ablation table (CAT only, MIL only, CATMIL) reports DSC, HD95, lesion F1,
small-lesion recall and false-positive volume, but not the number of completely
missed lesions. This script computes it from the stored 5-fold test predictions
with the definition of scripts/build_lesion_tables.py: 6-connected ground-truth
components, a lesion is missed when no voxel of the predicted mask (p > 0.5)
touches it. Counts are averaged over the 12 test scans x 5 fold models.
DiceCE (nnUNet) is included as a check against the lesion tables (4.95).

Usage: python ablation_fn_count.py
Output: analyses_2026-11/results/ablation_fn_count.csv
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = "/Volumes/BACH2TB/Tunka"
PRED = f"{ROOT}/SmallLesionMRI/MSLesSeg/final_checkpoints"
GT = f"{ROOT}/nnUNet_raw/Dataset333_MSLesSeg/labelsTs"
MODELS = {"DiceCE": "nnUNet", "CAT": "CAT", "MIL": "MIL", "CATMIL": "CATMIL"}
FOLDS = range(5)
STRUCT = ndimage.generate_binary_structure(3, 1)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "ablation_fn_count.csv")


def main():
    cases = sorted(f[:-7] for f in os.listdir(GT) if f.endswith(".nii.gz"))
    gts = {}
    for c in cases:
        gt = np.asarray(nib.load(f"{GT}/{c}.nii.gz").dataobj) > 0.5
        gts[c] = ndimage.label(gt, structure=STRUCT)

    rows = []
    for model, pred_dir in MODELS.items():
        for fold in FOLDS:
            for c in cases:
                lab, n = gts[c]
                seg = np.asarray(nib.load(f"{PRED}/{pred_dir}/fold_{fold}/{c}.nii.gz").dataobj) > 0
                covered = ndimage.sum_labels(seg, lab, np.arange(1, n + 1))
                rows.append({"model": model, "fold": fold, "case": c,
                             "gt_n": n, "fn_count": int((covered == 0).sum())})
    df = pd.DataFrame(rows)
    per_fold = df.groupby(["model", "fold"]).fn_count.mean().reset_index()
    summary = per_fold.groupby("model", sort=False).fn_count.mean().reindex(list(MODELS)).reset_index()
    summary.to_csv(OUT, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
