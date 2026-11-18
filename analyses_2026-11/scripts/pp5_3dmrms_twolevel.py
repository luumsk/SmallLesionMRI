"""3D-MR-MS after PP5: paired differences CATMIL - each loss, with the arXiv two-level bootstrap.

scripts/stats_significance.py cannot analyse PP5 on 3D-MR-MS, because the
Ljubljana *_pp5 metric files are copies of the unfiltered ones. This script
computes the per-scan lesion metrics from the lesion tables instead, with the
per-scan definitions of analyses_2026-11/scripts/architecture_vs_loss.per_scan
(6-connected components, any-voxel overlap, p > 0.5; PP5: predicted components
< 5 voxels removed, a lesion stays detected only if an overlapping component of
>= 5 voxels exists), and the bootstrap of stats_significance.py (patients and
fold models resampled jointly, 10,000 resamples). The unfiltered rows are a
check: they must reproduce metrics/stats/significance_case_level.csv.

Usage: python pp5_3dmrms_twolevel.py
Output: ../results/pp5_3dmrms_twolevel.csv
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
from stats_significance import SEED, boot_weights, patient_of, weighted_mean  # noqa: E402

TABLES = os.path.join(REPO, "metrics", "lesion_tables")
OUT = os.path.join(REPO, "analyses_2026-11", "results")
DATASET, SMALL_VOX, EPS = "3D-MR-MS", 150, 1e-8
KEYS = ["model", "fold", "case"]
METRICS = {"small_recall": True, "fn_count": False, "lesion_f1": True}  # True: higher is better
REFS = ["DiceCE", "Tversky", "FocalTversky", "CCDiceCE"]


def per_scan(les, comp, pp):
    if pp:
        les = les.assign(detected=les["hit_size_t0.5"] >= pp)
        comp = comp[comp.size_vox >= pp]
    les = les.assign(small=les.size_vox <= SMALL_VOX)
    les["small_hit"] = les.small & les.detected
    g = les.groupby(KEYS).agg(gt_n=("detected", "size"), tp_gt=("detected", "sum"),
                              small_n=("small", "sum"), small_hit=("small_hit", "sum"))
    p = comp.groupby(KEYS).agg(pr_n=("matched", "size"), tp_pr=("matched", "sum"))
    df = g.join(p).fillna(0)
    rec, prec = df.tp_gt / (df.gt_n + EPS), df.tp_pr / (df.pr_n + EPS)
    df["lesion_f1"] = 2 * prec * rec / (prec + rec + EPS)
    df["fn_count"] = df.gt_n - df.tp_gt
    df["small_recall"] = df.small_hit / (df.small_n + EPS)
    return df[list(METRICS)].reset_index()


def main():
    les = pd.read_csv(os.path.join(TABLES, "lesions.csv"))
    comp = pd.read_csv(os.path.join(TABLES, "components.csv"))
    les = les[les.dataset == DATASET]
    comp = comp[(comp.dataset == DATASET) & (comp.threshold == 0.5)]
    rows = []
    for pp, setting in ((0, "3D-MR-MS"), (5, "3D-MR-MS (pp5)")):
        df = per_scan(les, comp, pp)
        cases = sorted(df.case.unique(), key=lambda c: (patient_of(c), c))
        patients = [patient_of(c) for c in cases]
        for metric, higher in METRICS.items():
            x = {m: df[df.model == m].pivot(index="fold", columns="case", values=metric)
                 .reindex(index=range(5), columns=cases).to_numpy(float)
                 for m in ["CATMIL"] + REFS}
            for ref in REFS:
                w_scan, w_fold = boot_weights(patients, np.random.default_rng(SEED))
                boot = weighted_mean(x["CATMIL"], w_scan, w_fold) - weighted_mean(x[ref], w_scan, w_fold)
                lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                pa = pd.Series(np.nanmean(x["CATMIL"], 0), index=cases).groupby(patients).mean()
                pb = pd.Series(np.nanmean(x[ref], 0), index=cases).groupby(patients).mean()
                rows.append(dict(setting=setting, reference=ref, metric=metric,
                                 catmil_mean=np.nanmean(x["CATMIL"]), ref_mean=np.nanmean(x[ref]),
                                 delta=np.nanmean(x["CATMIL"]) - np.nanmean(x[ref]), ci_low=lo,
                                 ci_high=hi, patients_favouring_catmil=int(
                                     ((pa > pb) if higher else (pa < pb)).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "pp5_3dmrms_twolevel.csv"), index=False)
    with pd.option_context("display.width", 200):
        print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
