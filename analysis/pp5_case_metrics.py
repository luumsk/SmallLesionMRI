"""Per-case metrics after PP5 (remove predicted components < 5 voxels) on MSLesSeg.

The PP5 metrics of the original four losses were computed with multiscale_eval.py
on filtered masks saved as <model>_pp5/. CC-DiceCE has no filtered masks, so this
script filters the saved masks in memory (6-connected components, as in
multiscale_eval._label_cc) and calls multiscale_eval.compute_case_metrics in
"main" mode with its default arguments. The output has the same format as
multiscale_eval.py, so stats_significance.py reads it unchanged.

DiceCE (nnUNet) is recomputed as a check against the existing
multiscale_eval_nnUNet_pp5_fold*_main.csv; its output is written with a
"_check" suffix and never replaces the original files.

Usage: python pp5_case_metrics.py [n_workers]
Outputs: metrics_tunka/MSLesSeg/main/multiscale_eval_CCDiceCE_pp5_fold{k}_main.csv
         metrics_tunka/MSLesSeg/main/multiscale_eval_nnUNet_pp5_check_fold{k}_main.csv
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import multiscale_eval as me  # noqa: E402

ROOT = os.environ.get("SLS_DATA_ROOT", "/Volumes/BACH2TB/Tunka")  # see README, "Data location"
PRED = f"{ROOT}/SmallLesionMRI/MSLesSeg/final_checkpoints"
GT = f"{ROOT}/nnUNet_raw/Dataset333_MSLesSeg/labelsTs"
OUT = os.path.join(REPO, "metrics_tunka", "MSLesSeg", "main")
MIN_VOX = 5
MODELS = {"CCDiceCE": "CCDiceCE_pp5", "nnUNet": "nnUNet_pp5_check"}  # folder -> output name


def remove_small(mask):
    lab, n = me._label_cc(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = sizes >= MIN_VOX
    keep[0] = False
    return keep[lab]


def job(args):
    folder, fold, case = args
    gt, spacing = me.load_nii_mask(Path(f"{GT}/{case}.nii.gz"))
    pred, _ = me.load_nii_mask(Path(f"{PRED}/{folder}/fold_{fold}/{case}.nii.gz"))
    metrics = me.compute_case_metrics(
        gt=gt, pred=remove_small(pred), spacing=spacing, prob=None, small_voxels_thresh=150,
        ece_bins=15, ece_max_points=200000, ece_seed=123, fp_near_mm=2.0, fp_far_mm=5.0,
        metrics_mode="main")
    return folder, fold, {"case_id": case, **metrics}


def main(workers=4):
    cases = sorted(f[:-7] for f in os.listdir(GT) if f.endswith(".nii.gz"))
    jobs = [(m, k, c) for m in MODELS for k in range(5) for c in cases]
    with Pool(workers) as p:
        results = p.map(job, jobs, chunksize=1)
    for folder, name in MODELS.items():
        for k in range(5):
            rows = [r for f, fk, r in results if f == folder and fk == k]
            path = os.path.join(OUT, f"multiscale_eval_{name}_fold{k}_main.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            print("wrote", path)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
