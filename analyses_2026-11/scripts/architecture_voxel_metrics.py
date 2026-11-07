"""Dice and HD95 for the architecture-vs-loss comparison (MSLesSeg test set).

Same protocol as the Ch3 tables: per fold model, Dice and HD95 (mm) of every
test scan with multiscale_eval.dice_score / hd95_mm (saved masks, p > 0.5), mean
over the 12 scans, then mean +- SD over the fold models. nnU-Net DiceCE and
CATMIL are recomputed as a check against the dissertation (Dice 0.7802 / 0.783,
HD95 9.04 / 7.98 mm).

Usage: python architecture_voxel_metrics.py [n_workers]
Outputs: ../results/architecture_voxel_metrics_{per_case,summary}.csv
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
import multiscale_eval as me  # noqa: E402

ROOT = "/Volumes/BACH2TB/Tunka"
PRED = f"{ROOT}/SmallLesionMRI/MSLesSeg/final_checkpoints"
GT = f"{ROOT}/nnUNet_raw/Dataset333_MSLesSeg/labelsTs"
OUT = os.path.join(REPO, "analyses_2026-11", "results")
MODELS = {"nnU-Net (CATMIL)": "CATMIL", "nnU-Net (DiceCE)": "nnUNet", "SegResNet (DiceCE)": "SegResNet",
          "SwinUNETR (DiceCE)": "SwinUNETR", "UNETR (DiceCE)": "UNETR", "UMambaBot (DiceCE)": "UMambaBot"}


def job(args):
    name, folder, fold, case = args
    gt, spacing = me.load_nii_mask(Path(f"{GT}/{case}.nii.gz"))
    pr, _ = me.load_nii_mask(Path(f"{PRED}/{folder}/fold_{fold}/{case}.nii.gz"))
    return {"model": name, "fold": fold, "case": case, "dice": me.dice_score(gt, pr),
            "hd95_mm": me.hd95_mm(gt, pr, spacing)}


def main(workers=4):
    cases = sorted(f[:-7] for f in os.listdir(GT) if f.endswith(".nii.gz"))
    jobs = [(n, f, k, c) for n, f in MODELS.items() for k in range(5) for c in cases
            if os.path.exists(f"{PRED}/{f}/fold_{k}/{c}.nii.gz")]
    with Pool(workers) as p:
        rows = p.map(job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "architecture_voxel_metrics_per_case.csv"), index=False)
    n_inf = int(np.isinf(df.hd95_mm).sum())
    per_fold = df.replace(np.inf, np.nan).groupby(["model", "fold"])[["dice", "hd95_mm"]].mean()
    summ = per_fold.groupby("model").agg(["mean", "std", "count"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.loc[list(MODELS)]
    summ.to_csv(os.path.join(OUT, "architecture_voxel_metrics_summary.csv"))
    print(f"infinite HD95 values (empty prediction or reference): {n_inf}")
    print(summ.round(4).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
