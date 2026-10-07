"""Statistical significance of CATMIL vs the baseline losses.

Two analyses, both treating the patient as the independent unit (MSLesSeg test scans
are longitudinal: 12 scans from 6 patients) and keeping the fold models paired (fold k
of every loss was trained on the same split).

1. Case-level metrics (per-fold per-case CSVs from multiscale_eval.py):
   - Effect: mean difference CATMIL - baseline, computed like the paper table
     (mean over scans of the five-fold mean).
   - 95% CI: two-level bootstrap, resampling patients (with all their scans) and fold
     indices (jointly for both losses).
   - p-value: exact Wilcoxon signed-rank test on patient-level differences, Holm-corrected
     over the baselines within each metric. With 6 patients the smallest attainable
     two-sided p is 1/32 = 0.031.

2. Lesion-level detection (metrics/lesion_tables/lesions.csv), by size bin:
   - Effect: pooled recall over all GT lesions and folds, CATMIL - baseline.
   - 95% CI: the same two-level patient/fold bootstrap.
   - p-value: exact patient-level permutation test (swap the two losses' labels within
     patients, all 2^n_patients assignments).
   - Odds ratio: GEE logistic model, clustered by patient, bias-reduced sandwich variance.

Usage: python stats_significance.py <metrics_dir> <lesion_table_csv> <out_dir>
  e.g. python stats_significance.py metrics_tunka metrics/lesion_tables/lesions.csv metrics/stats
"""
import itertools
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

N_BOOT = 10000
N_FOLDS = 5
SEED = 0
METRICS = {  # name -> True if higher is better
    "small_lesion_recall": True,
    "fn_lesion_count": False,
    "fn_volume_fraction": False,
    "dice": True,
    "hd95_mm": False,
    "lesion_f1": True,
    "fp_volume_mm3": False,
}
# (dataset dir, split, file suffix, display name)
SETTINGS = [
    ("MSLesSeg", "main", "", "MSLesSeg"),
    ("MSLesSeg", "main", "_pp5", "MSLesSeg (pp5)"),
    ("Ljubljana", "all", "", "3D-MR-MS"),
    # Ljubljana *_pp5 files are copies of the unfiltered ones (all but CATMIL fold 0), so
    # PP5 on 3D-MR-MS is analysed from the lesion tables (operating_curves.py) instead.
]
# family name -> list of (method, reference); each family is Holm-corrected separately
FAMILIES = {
    "main": [("CATMIL", "nnUNet"), ("CATMIL", "Tversky"), ("CATMIL", "FocalTversky")],
    "ablation": [("CATMIL", "CAT"), ("CATMIL", "MIL"), ("CAT", "nnUNet"), ("MIL", "nnUNet")],
}
LESION_BINS = {  # name -> (min voxels exclusive, max voxels inclusive)
    "small (<=150)": (0, 150),
    "tiny (<=10)": (0, 10),
    "11-50": (10, 50),
    "51-200": (50, 200),
    ">200": (200, np.inf),
    "all": (0, np.inf),
}
LESION_MODEL_NAMES = {"nnUNet": "DiceCE"}


def patient_of(case_id):
    return case_id.split("_")[0]


def load_cases(metrics_dir, dataset, split, model, suffix):
    """Return (metric -> array[fold, scan]), case ids, patient ids."""
    frames = []
    for fold in range(N_FOLDS):
        path = f"{metrics_dir}/{dataset}/{split}/multiscale_eval_{model}{suffix}_fold{fold}_{split}.csv"
        if not os.path.exists(path):
            return None
        frames.append(pd.read_csv(path).sort_values("case_id").reset_index(drop=True))
    cases = frames[0]["case_id"].tolist()
    assert all(f["case_id"].tolist() == cases for f in frames)
    values = {m: np.stack([f[m].to_numpy(float) for f in frames]) for m in METRICS}
    return values, cases, [patient_of(c) for c in cases]


def boot_weights(patients, rng):
    """Bootstrap weights: (B, n_scans) from resampled patients, (B, n_folds) from resampled folds."""
    uniq = sorted(set(patients))
    scan_patient = np.array([uniq.index(p) for p in patients])
    pat_counts = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=N_BOOT)
    fold_counts = rng.multinomial(N_FOLDS, np.full(N_FOLDS, 1 / N_FOLDS), size=N_BOOT)
    return pat_counts[:, scan_patient].astype(float), fold_counts.astype(float)


def weighted_mean(x, w_scan, w_fold):
    """Bootstrap replicates of the NaN-aware mean of x[fold, scan]."""
    valid = ~np.isnan(x)
    num = np.einsum("bf,bs,fs->b", w_fold, w_scan, np.where(valid, x, 0.0))
    den = np.einsum("bf,bs,fs->b", w_fold, w_scan, valid.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def patient_means(x, patients):
    """Fold-mean per scan, then mean per patient."""
    per_scan = np.nanmean(x, axis=0)
    return pd.Series(per_scan).groupby(np.array(patients)).mean()


def case_level(metrics_dir):
    rows = []
    for dataset, split, suffix, name in SETTINGS:
        cache = {}
        for family, pairs in FAMILIES.items():
            for method, ref in pairs:
                for model in (method, ref):
                    if model not in cache:
                        cache[model] = load_cases(metrics_dir, dataset, split, model, suffix)
                if cache[method] is None or cache[ref] is None:
                    continue
                (xa, cases, patients), (xb, cases_b, _) = cache[method], cache[ref]
                assert cases == cases_b
                rng = np.random.default_rng(SEED)
                w_scan, w_fold = boot_weights(patients, rng)
                for metric, higher_better in METRICS.items():
                    a, b = xa[metric], xb[metric]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        mean_a = np.nanmean(np.nanmean(a, 0))
                        mean_b = np.nanmean(np.nanmean(b, 0))
                        boot = weighted_mean(a, w_scan, w_fold) - weighted_mean(b, w_scan, w_fold)
                        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
                        d_pat = (patient_means(a, patients) - patient_means(b, patients)).dropna()
                    p = wilcoxon(d_pat, method="exact").pvalue if (d_pat != 0).any() else 1.0
                    delta = mean_a - mean_b
                    rows.append(dict(
                        setting=name, family=family, method=method, reference=ref, metric=metric,
                        method_mean=mean_a, reference_mean=mean_b, delta=delta, ci_low=lo, ci_high=hi,
                        n_patients=len(d_pat), patients_favouring_method=int(
                            ((d_pat > 0) if higher_better else (d_pat < 0)).sum()),
                        method_better=(delta > 0) == higher_better, p_wilcoxon=p))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, idx in df.groupby(["setting", "family", "metric"]).groups.items():
        df.loc[idx, "p_holm"] = multipletests(df.loc[idx, "p_wilcoxon"], method="holm")[1]
    return df


def lesion_level(lesion_csv):
    les = pd.read_csv(lesion_csv)
    les["patient"] = les["case"].map(patient_of)
    les["detected"] = les["detected"].astype(float)
    rows = []
    for dataset, sub in les.groupby("dataset"):
        for _, ref in FAMILIES["main"]:
            ref_name = LESION_MODEL_NAMES.get(ref, ref)
            for bin_name, (lo_v, hi_v) in LESION_BINS.items():
                s = sub[(sub["size_vox"] > lo_v) & (sub["size_vox"] <= hi_v)]
                # det[model] -> array[fold, lesion]; lesions identified by (case, lesion_id)
                det, keys = {}, None
                for model in ("CATMIL", ref_name):
                    piv = s[s["model"] == model].pivot_table(
                        index=["case", "lesion_id"], columns="fold", values="detected")
                    piv = piv.sort_index()
                    keys = piv.index if keys is None else keys
                    assert piv.index.equals(keys)
                    det[model] = piv.to_numpy().T
                if keys is None or len(keys) == 0:
                    continue
                patients = [patient_of(c) for c, _ in keys]
                a, b = det["CATMIL"], det[ref_name]
                rng = np.random.default_rng(SEED)
                w_scan, w_fold = boot_weights(patients, rng)
                boot = weighted_mean(a, w_scan, w_fold) - weighted_mean(b, w_scan, w_fold)
                ci = np.nanpercentile(boot, [2.5, 97.5])
                delta = a.mean() - b.mean()
                rows.append(dict(
                    dataset=dataset, reference=ref, size_bin=bin_name, n_lesions=len(keys),
                    n_patients=len(set(patients)), recall_catmil=a.mean(), recall_reference=b.mean(),
                    delta=delta, ci_low=ci[0], ci_high=ci[1],
                    p_permutation=cluster_permutation_p(a, b, patients, delta),
                    **gee_odds_ratio(a, b, patients)))
    df = pd.DataFrame(rows)
    df["p_holm"] = np.nan
    for _, idx in df.groupby(["dataset", "size_bin"]).groups.items():
        df.loc[idx, "p_holm"] = multipletests(df.loc[idx, "p_permutation"], method="holm")[1]
    return df


def cluster_permutation_p(a, b, patients, observed):
    """Exact two-sided p: swap the two losses within each patient, over all 2^n assignments."""
    patients = np.array(patients)
    uniq = sorted(set(patients))
    hits_a = np.array([a[:, patients == p].sum() for p in uniq])
    hits_b = np.array([b[:, patients == p].sum() for p in uniq])
    total = a.size
    stats = []
    for flips in itertools.product([1, -1], repeat=len(uniq)):
        f = np.array(flips)
        stats.append(((hits_a - hits_b) * f).sum() / total)
    stats = np.array(stats)
    return float(np.mean(np.abs(stats) >= abs(observed) - 1e-12))


def gee_odds_ratio(a, b, patients):
    """Odds ratio of detection for CATMIL vs reference, GEE clustered by patient."""
    n_folds, n_les = a.shape
    y = np.concatenate([a.ravel(), b.ravel()])
    x = np.concatenate([np.ones(a.size), np.zeros(b.size)])
    groups = np.tile(np.repeat(np.array(patients)[None, :], n_folds, axis=0).ravel(), 2)
    if y.min() == y.max():
        return dict(odds_ratio=np.nan, or_ci_low=np.nan, or_ci_high=np.nan, p_gee=np.nan)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GEE(y, sm.add_constant(x), groups=groups, family=sm.families.Binomial(),
                         cov_struct=sm.cov_struct.Exchangeable()).fit(cov_type="bias_reduced")
        lo, hi = fit.conf_int()[1]
        return dict(odds_ratio=np.exp(fit.params[1]), or_ci_low=np.exp(lo), or_ci_high=np.exp(hi),
                    p_gee=fit.pvalues[1])
    except Exception:  # perfect separation etc.
        return dict(odds_ratio=np.nan, or_ci_low=np.nan, or_ci_high=np.nan, p_gee=np.nan)


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def print_case_table(df):
    for (setting, family), g in df.groupby(["setting", "family"], sort=False):
        print(f"\n=== {setting} — {family} (case-level; Δ = method − reference) ===")
        for _, r in g.iterrows():
            flag = "*" if r.p_holm < 0.05 else " "
            print(f"{r.method:>7} vs {r.reference:<12} {r.metric:<20} "
                  f"Δ={r.delta:+9.4f} [{r.ci_low:+9.4f}, {r.ci_high:+9.4f}]  "
                  f"{'better' if r.method_better else 'worse ':<6} "
                  f"{r.patients_favouring_method}/{r.n_patients} pts  "
                  f"p={fmt_p(r.p_wilcoxon)} p_holm={fmt_p(r.p_holm)}{flag}")


def print_lesion_table(df):
    for dataset, g in df.groupby("dataset"):
        print(f"\n=== {dataset} — lesion-level recall, CATMIL − reference ===")
        for _, r in g.iterrows():
            flag = "*" if r.p_holm < 0.05 else " "
            print(f"vs {r.reference:<12} {r.size_bin:<13} n={r.n_lesions:>3}  "
                  f"{r.recall_catmil:.3f} vs {r.recall_reference:.3f}  "
                  f"Δ={r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]  "
                  f"OR={r.odds_ratio:.2f} [{r.or_ci_low:.2f}, {r.or_ci_high:.2f}] p_gee={fmt_p(r.p_gee)}  "
                  f"p_perm={fmt_p(r.p_permutation)} p_holm={fmt_p(r.p_holm)}{flag}")


def main(metrics_dir, lesion_csv, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cases = case_level(metrics_dir)
    cases.to_csv(os.path.join(out_dir, "significance_case_level.csv"), index=False)
    print_case_table(cases)
    lesions = lesion_level(lesion_csv)
    lesions.to_csv(os.path.join(out_dir, "significance_lesion_level.csv"), index=False)
    print_lesion_table(lesions)


if __name__ == "__main__":
    main(*sys.argv[1:4])
