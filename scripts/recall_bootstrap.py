"""Patient-level bootstrap of lesion-level metrics on both test sets.

Reads metrics/lesion_tables/lesions.csv and components.csv (written by
build_lesion_tables.py) and recomputes, as in multiscale_eval.py, per scan:
  small_recall  recall of reference lesions <= 150 voxels
  fn_count      reference lesions with no overlapping predicted component
  lesion_f1     F1 of lesion-wise recall and component-wise precision
Each metric is averaged over scans and then over the five fold models, which
reproduces the reported values (e.g. CATMIL small-lesion recall 0.8730, FN
count 2.57, lesion F1 0.7571 on MSLesSeg). With --pp5, predicted components
smaller than 5 voxels are removed first (PP5; lesion F1 0.8415 for CATMIL).

The test patients are resampled with replacement (all scans of a patient are
kept together) and the paired difference CATMIL - baseline is recomputed for
each resample; 95% percentile intervals are printed.

Usage: python recall_bootstrap.py [--pp5] [n_resamples]
"""
import os
import sys

import numpy as np
import pandas as pd

TABLES = os.path.join(os.path.dirname(__file__), os.pardir, "metrics", "lesion_tables")
LOSSES = ["CATMIL", "DiceCE", "Tversky", "FocalTversky"]
METRICS = ["small_recall", "fn_count", "lesion_f1"]
SMALL_VOX = 150
EPS = 1e-8
KEYS = ["dataset", "model", "fold", "case"]


def per_scan_metrics(pp_min_vox=0):
    """One row per (dataset, loss, fold, scan) with the three metrics."""
    les = pd.read_csv(os.path.join(TABLES, "lesions.csv"))
    comp = pd.read_csv(os.path.join(TABLES, "components.csv"))
    comp = comp[comp.threshold == 0.5]  # the saved nnU-Net masks are p > 0.5
    if pp_min_vox:
        # a reference lesion stays detected only if an overlapping component survives the filter
        les = les.assign(detected=les["hit_size_t0.5"] >= pp_min_vox)
        comp = comp[comp.size_vox >= pp_min_vox]
    les["small"] = les.size_vox <= SMALL_VOX
    les["small_hit"] = les.small & les.detected
    g = les.groupby(KEYS).agg(gt_n=("detected", "size"), tp_gt=("detected", "sum"),
                              small_n=("small", "sum"), small_hit=("small_hit", "sum"))
    p = comp.groupby(KEYS).agg(pr_n=("matched", "size"), tp_pr=("matched", "sum"))
    df = g.join(p).fillna(0)
    recall = df.tp_gt / (df.gt_n + EPS)
    precision = df.tp_pr / (df.pr_n + EPS)
    df["lesion_f1"] = 2 * precision * recall / (precision + recall + EPS)
    df["fn_count"] = df.gt_n - df.tp_gt
    df["small_recall"] = df.small_hit / (df.small_n + EPS)
    return df[METRICS].reset_index()


def main(pp5=False, n_resamples=2000):
    df = per_scan_metrics(5 if pp5 else 0)
    lo, hi = int(0.025 * n_resamples), int(0.975 * n_resamples) - 1
    rng = np.random.default_rng(0)
    for ds in ["MSLesSeg", "3D-MR-MS"]:
        d = df[df.dataset == ds].copy()
        # MSLesSeg cases are timepoints (P13_T1, P13_T2, ...) of one patient
        d["patient"] = d.case.str.split("_").str[0] if ds == "MSLesSeg" else d.case
        # metric per (loss, patient-scan) averaged over folds; scans weighted equally
        per_case = d.groupby(["model", "patient", "case"])[METRICS].mean()
        patients = sorted(d.patient.unique())
        cases_of = {p: sorted(d[d.patient == p].case.unique()) for p in patients}

        def value(loss, cases):
            return per_case.loc[loss].reset_index(level=0, drop=True).loc[cases].mean()

        all_cases = [c for p in patients for c in cases_of[p]]
        boot = []
        for _ in range(n_resamples):
            cases = [c for p in rng.choice(patients, len(patients)) for c in cases_of[p]]
            boot.append({m: value(m, cases) for m in LOSSES})
        print(f"{ds}{' (PP5)' if pp5 else ''}: {len(patients)} patients, {len(all_cases)} scans")
        for metric in METRICS:
            point = {m: value(m, all_cases)[metric] for m in LOSSES}
            print(f"  {metric}: " + ", ".join(f"{m} {point[m]:.4f}" for m in LOSSES))
            for m in LOSSES[1:]:
                diff = sorted(b["CATMIL"][metric] - b[m][metric] for b in boot)
                print(f"    CATMIL - {m:13s} {point['CATMIL'] - point[m]:+.4f}  "
                      f"95% CI [{diff[lo]:+.4f}, {diff[hi]:+.4f}]")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--pp5"]
    main("--pp5" in sys.argv, int(args[0]) if args else 2000)
