"""Does loss of the periventricular (PV) indicator follow the independence model r^k?

Under independence, a scan whose indicator rests on k reference PV lesions loses it
(all k missed) with probability prod_i r(size_i), where r(size) is the per-lesion miss
rate of the size bin. This script tests that model on the Ch3 test scans
(MSLesSeg 12 scans, 3D-MR-MS 6 scans), for each loss and fold model, D = 3 mm, raw masks.

Inputs: metrics/periventricular/pv_lesions.csv and pv_scans.csv (periventricular_flip.py).
  1. Reproduce Table tab:pv_missed (miss rates of all / small PV lesions).
  2. Expected number of lost indicators under independence (size-aware r, leave-one-scan-out
     estimate) vs the observed number (all reference PV lesions of a scan missed).
  3. Clustering: Pearson dispersion of the number of missed PV lesions per scan
     (observed / Poisson-binomial variance), with a permutation test that shuffles miss labels
     across scans within (dataset, loss, fold, size bin); and the pairwise co-miss ratio
     P(both missed) / (r_i r_j) for lesion pairs in the same scan.

Usage: python -W ignore pv_independence_check.py
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(REPO, "analyses_2026-11", "results")
D_MM = 3
BINS = [0, 150, 500, np.inf]  # voxels: small (<=150, as in Ch3), medium, large
N_PERM = 2000
rng = np.random.default_rng(0)

les = pd.read_csv(os.path.join(REPO, "metrics/periventricular/pv_lesions.csv"))
les = les[les.dist_to_ventricle_mm <= D_MM].copy()
les["missed"] = ~les.detected
les["bin"] = pd.cut(les.size_vox, BINS, labels=False, include_lowest=True)
les["small"] = les.size_vox <= 150

# 1. reproduce tab:pv_missed (mean over folds of the per-fold fraction == pooled, equal counts per fold)
rep = les.groupby(["dataset", "loss"]).apply(
    lambda g: pd.Series({"all": g.missed.mean(), "small": g[g.small].missed.mean(),
                         "n_lesions_per_fold": len(g) / g.fold.nunique(),
                         "n_small_per_fold": g.small.sum() / g.fold.nunique()})).round(3)
print("Reproduction of tab:pv_missed\n", rep, "\n")


def rates(g):
    """Miss rate per size bin (pooled over the rows of g)."""
    return g.groupby("bin").missed.mean()


rows, disp_rows, pair_rows = [], [], []
for (ds, loss), g in les.groupby(["dataset", "loss"]):
    cases = sorted(g.case.unique())
    for fold, gf in g.groupby("fold"):
        r_full = rates(gf)
        exp_lost = obs_lost = 0.0
        for c in cases:
            gc = gf[gf.case == c]
            # leave-one-scan-out rate (pooled over the other scans of this fold); fall back to the
            # rate of all folds of this loss and dataset if the bin is empty elsewhere
            r_loso = rates(gf[gf.case != c])
            r_alt = rates(g[g.case != c])
            r_i = np.array([r_loso.get(b, r_alt.get(b, r_full.get(b))) for b in gc.bin])
            p_lost = float(np.prod(r_i))
            lost = bool(gc.missed.all())
            exp_lost += p_lost
            obs_lost += lost
            rows.append({"dataset": ds, "loss": loss, "fold": fold, "case": c, "k": len(gc),
                         "k_small": int(gc.small.sum()), "missed": int(gc.missed.sum()),
                         "p_lost_indep": p_lost, "lost_obs": lost})

        # 3a. dispersion of missed counts per scan (full-data size-specific rates)
        def dispersion(miss_series):
            num = den = 0.0
            for c in cases:
                m = miss_series[gf.case.values == c]
                r = gf.bin[gf.case.values == c].map(r_full).values
                num += (m.sum() - r.sum()) ** 2
                den += (r * (1 - r)).sum()
            return num / den if den > 0 else np.nan

        obs_d = dispersion(gf.missed.values)
        perm = []
        miss = gf.missed.values.copy()
        bins = gf.bin.values
        for _ in range(N_PERM):
            sh = miss.copy()
            for b in np.unique(bins):
                idx = np.flatnonzero(bins == b)
                sh[idx] = rng.permutation(sh[idx])
            perm.append(dispersion(sh))
        perm = np.array(perm)
        disp_rows.append({"dataset": ds, "loss": loss, "fold": fold, "dispersion_obs": obs_d,
                          "dispersion_perm_mean": float(np.nanmean(perm)),
                          "p_perm": float((np.sum(perm >= obs_d) + 1) / (N_PERM + 1))})

        # 3b. pairwise co-miss within scans
        both = expect = 0.0
        for c in cases:
            gc = gf[gf.case == c]
            m = gc.missed.values.astype(float)
            r = gc.bin.map(r_full).values
            s_m, s_r = m.sum(), r.sum()
            both += (s_m ** 2 - (m ** 2).sum()) / 2
            expect += (s_r ** 2 - (r ** 2).sum()) / 2
        pair_rows.append({"dataset": ds, "loss": loss, "fold": fold, "pairs_both_missed": both,
                          "pairs_expected_indep": expect,
                          "co_miss_ratio": both / expect if expect > 0 else np.nan})

scans = pd.DataFrame(rows)
disp = pd.DataFrame(disp_rows)
pairs = pd.DataFrame(pair_rows)

# 2. expected vs observed lost indicators, per fold model, then mean over folds
lost = scans.groupby(["dataset", "loss", "fold"]).agg(expected=("p_lost_indep", "sum"),
                                                       observed=("lost_obs", "sum")).reset_index()
lost_sum = lost.groupby(["dataset", "loss"])[["expected", "observed"]].mean().round(3)
print("Lost indicators per fold model (mean over 5 folds): expected under independence vs observed\n",
      lost_sum, "\n")

single = scans[scans.k == 1]
print("Scans with k = 1\n", single.groupby(["dataset", "case", "loss"]).agg(
    p_lost_indep=("p_lost_indep", "mean"), lost_obs=("lost_obs", "mean")).round(3), "\n")

disp_sum = disp.groupby(["dataset", "loss"]).agg(
    dispersion_obs=("dispersion_obs", "mean"), dispersion_perm=("dispersion_perm_mean", "mean"),
    p_perm_median=("p_perm", "median"), folds_p_lt_05=("p_perm", lambda p: int((p < 0.05).sum()))).round(3)
print("Dispersion of missed PV lesions per scan (1 = independence)\n", disp_sum, "\n")
pair_sum = pairs.groupby(["dataset", "loss"])[["pairs_both_missed", "pairs_expected_indep",
                                                "co_miss_ratio"]].mean().round(2)
print("Pairwise co-miss within scans (ratio 1 = independence)\n", pair_sum, "\n")

kdist = scans.drop_duplicates(["dataset", "case"])[["dataset", "case", "k", "k_small"]]
print("k per scan\n", kdist.sort_values(["dataset", "k"]).to_string(index=False))

os.makedirs(OUT, exist_ok=True)
scans.to_csv(os.path.join(OUT, "pv_independence_scans.csv"), index=False)
disp.to_csv(os.path.join(OUT, "pv_independence_dispersion.csv"), index=False)
pairs.to_csv(os.path.join(OUT, "pv_independence_pairs.csv"), index=False)
summary = {
    "reproduction": rep.reset_index().to_dict("records"),
    "lost_indicators": lost_sum.reset_index().to_dict("records"),
    "dispersion": disp_sum.reset_index().to_dict("records"),
    "co_miss": pair_sum.reset_index().to_dict("records"),
    "k_per_scan": kdist.to_dict("records"),
    "settings": {"d_mm": D_MM, "size_bins_vox": [0, 150, 500, "inf"], "n_perm": N_PERM, "masks": "raw"},
}
with open(os.path.join(OUT, "pv_independence_summary.json"), "w") as f:
    json.dump(summary, f, indent=1, default=float)
