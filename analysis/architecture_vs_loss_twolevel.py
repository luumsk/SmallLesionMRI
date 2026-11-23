"""Architecture vs objective on the MSLesSeg test set with the arXiv paper's two-level bootstrap.

architecture_vs_loss.py resamples patients only (2000 resamples, as in the
dissertation). The arXiv paper (paper/arxiv/sections_v02) resamples patients and
fold models jointly (analysis/stats_significance.py: 10,000 resamples, fold k of
every model drawn together), so this script recomputes the paired differences
of the architecture comparison with that bootstrap. Per-scan metrics are taken
unchanged from architecture_vs_loss.per_scan (lesion tables) and
architecture_voxel_metrics_per_case.csv (DSC, HD95).

Comparisons: CATMIL nnU-Net - each architecture trained with DiceCE (change of
objective) and each architecture with DiceCE - DiceCE nnU-Net (change of
architecture), without and with PP5.

Usage: python architecture_vs_loss_twolevel.py
Output: results/analyses_2026-11/architecture_vs_loss_bootstrap_twolevel.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, HERE)
import architecture_vs_loss as avl  # noqa: E402
from stats_significance import SEED, boot_weights, patient_of, weighted_mean  # noqa: E402

ARCHS = ["SegResNet", "UMambaBot", "SwinUNETR", "UNETR"]
CATMIL, DICECE = "nnU-Net (CATMIL)", "nnU-Net (DiceCE)"
MODELS = [CATMIL, DICECE] + [f"{a} (DiceCE)" for a in ARCHS]


def per_scan(pp):
    old = avl.per_scan(avl.TABLES_OLD, pp)
    old = old[old.model.isin(["CATMIL", "DiceCE"])].replace(
        {"model": {"DiceCE": DICECE, "CATMIL": CATMIL}})
    new = avl.per_scan(avl.TABLES_NEW, pp)
    new["model"] = new.model + " (DiceCE)"
    df = pd.concat([old, new])
    if not pp:  # voxel metrics are only reported without post-processing
        vox = pd.read_csv(os.path.join(avl.OUT, "architecture_voxel_metrics_per_case.csv"))
        df = df.merge(vox, on=["model", "fold", "case"], how="left")
    return df


def arrays(df, metric, cases):
    """x[model] -> array[fold, scan]."""
    out = {}
    for m in MODELS:
        p = df[df.model == m].pivot(index="fold", columns="case", values=metric)
        out[m] = p.reindex(index=range(5), columns=cases).to_numpy(float)
    return out


def main():
    rows = []
    for pp, setting in ((0, "MSLesSeg"), (5, "MSLesSeg (pp5)")):
        df = per_scan(pp)
        cases = sorted(df.case.unique(), key=lambda c: (patient_of(c), c))
        patients = [patient_of(c) for c in cases]
        metrics = avl.METRICS + (["dice", "hd95_mm"] if not pp else [])
        for metric in metrics:
            x = arrays(df, metric, cases)
            higher_better = metric not in ("fn_count", "hd95_mm")
            for kind, a, b in ([("objective", CATMIL, m) for m in MODELS[2:]]
                               + [("architecture", m, DICECE) for m in MODELS[2:]]):
                w_scan, w_fold = boot_weights(patients, np.random.default_rng(SEED))
                boot = weighted_mean(x[a], w_scan, w_fold) - weighted_mean(x[b], w_scan, w_fold)
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                pa = pd.Series(np.nanmean(x[a], 0), index=cases).groupby(patients).mean()
                pb = pd.Series(np.nanmean(x[b], 0), index=cases).groupby(patients).mean()
                better = int(((pa > pb) if higher_better else (pa < pb)).sum())
                rows.append(dict(setting=setting, comparison=kind, a=a, b=b, metric=metric,
                                 a_mean=np.nanmean(x[a]), b_mean=np.nanmean(x[b]),
                                 delta=np.nanmean(x[a]) - np.nanmean(x[b]), ci_low=lo, ci_high=hi,
                                 n_patients=len(set(patients)), patients_favouring_a=better))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(avl.OUT, "architecture_vs_loss_bootstrap_twolevel.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(out.drop(columns=["n_patients"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
