"""Which lesions are missed, by which loss, and how close the misses are.

Uses lesions.csv from build_lesion_tables.py (default segmentation, p > 0.5) and
lesion_extent.csv from lesion_extent.py. A lesion
counts as detected by a loss if at least 3 of the 5 fold models detect it.

Outputs
  rescue_by_size.csv      per size bin: lesions both losses find, CATMIL only, reference
                          only, neither (CATMIL vs each baseline)
  miss_proximity.csv      missed lesion-fold instances by max foreground probability inside
                          the lesion: <0.1 (no signal), 0.1-0.3 (weak), 0.3-0.5 (near miss)
  missed_by_all.csv       lesions every loss misses: size, isolation, max probability
  isolation.csv           recall of small lesions by distance to the nearest other lesion
  clinical_size.csv       recall below / above the MAGNIMS 3 mm size (largest in-plane
                          diameter, from lesion_extent.py)
  fold_stability.csv      lesions found by only 1-4 of the 5 fold models (unstable)
  recall_vs_size.{pdf,jpeg}, miss_proximity.{pdf,jpeg}

Usage: python error_analysis.py <lesion_table_dir> <out_dir>
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from stats_significance import SEED, boot_weights, patient_of, weighted_mean

MODELS = ["CATMIL", "DiceCE", "Tversky", "FocalTversky"]
DATASETS = ["MSLesSeg", "3D-MR-MS"]
SIZE_BINS = [(0, 10, "≤10"), (10, 50, "11–50"), (50, 150, "51–150"), (150, 200, "151–200"),
             (200, np.inf, ">200")]
PROB_BINS = [(0, 0.1, "no signal (<0.1)"), (0.1, 0.3, "weak (0.1–0.3)"), (0.3, 0.5, "near miss (0.3–0.5)")]
ISOLATION_BINS = [(0, 5, "≤5 mm"), (5, 10, "5–10 mm"), (10, 20, "10–20 mm"), (20, np.inf, ">20 mm")]
SMALL_VOXELS = 150
CLINICAL_DIAMETER_MM = 3.0
STYLE = {"CATMIL": ("#e3a000", "o", "-"), "DiceCE": ("#404040", "s", "--"),
         "Tversky": ("#808080", "^", "-."), "FocalTversky": ("#a8a8a8", "D", ":")}
KEY = ["dataset", "case", "lesion_id"]


def size_bin(v):
    for lo, hi, name in SIZE_BINS:
        if lo < v <= hi:
            return name


def per_lesion(les):
    """One row per GT lesion: size, isolation, and per-loss detection rate / max probability."""
    les = les.copy()
    les["spacing_mm"] = np.cbrt(les["size_mm3"] / les["size_vox"])
    base = les[les.model == MODELS[0]].groupby(KEY).agg(
        size_vox=("size_vox", "first"), size_mm3=("size_mm3", "first"), cx=("cx", "first"),
        cy=("cy", "first"), cz=("cz", "first"), spacing_mm=("spacing_mm", "first")).reset_index()
    base["nearest_lesion_mm"] = np.nan
    for (_, _), g in base.groupby(["dataset", "case"]):
        pts = g[["cx", "cy", "cz"]].to_numpy() * g["spacing_mm"].to_numpy()[:, None]
        if len(pts) > 1:
            d, _ = cKDTree(pts).query(pts, k=2)
            base.loc[g.index, "nearest_lesion_mm"] = d[:, 1]
    agg = les.groupby(KEY + ["model"]).agg(det_rate=("detected", "mean"), max_prob=("max_prob", "mean"))
    agg = agg.unstack("model")
    agg.columns = [f"{a}_{m}" for a, m in agg.columns]
    out = base.merge(agg.reset_index(), on=KEY)
    for m in MODELS:
        out[f"det_{m}"] = out[f"det_rate_{m}"] >= 0.5  # >= 3 of 5 folds
    out["size_bin"] = out["size_vox"].map(size_bin)
    return out


def rescue_by_size(pl):
    rows = []
    for dataset in DATASETS:
        d = pl[pl.dataset == dataset]
        for ref in MODELS[1:]:
            for _, _, name in SIZE_BINS + [(0, 0, "all")]:
                g = d if name == "all" else d[d.size_bin == name]
                a, b = g["det_CATMIL"], g[f"det_{ref}"]
                rows.append(dict(dataset=dataset, reference=ref, size_bin=name, n=len(g),
                                 both=int((a & b).sum()), catmil_only=int((a & ~b).sum()),
                                 reference_only=int((~a & b).sum()), neither=int((~a & ~b).sum())))
    return pd.DataFrame(rows)


def miss_proximity(les):
    miss = les[~les.detected].copy()
    miss["small"] = miss.size_vox <= SMALL_VOXELS
    rows = []
    for (dataset, model), g in miss.groupby(["dataset", "model"]):
        n_folds = les[(les.dataset == dataset) & (les.model == model)].fold.nunique()
        row = dict(dataset=dataset, model=model, missed_per_fold=len(g) / n_folds)
        for lo, hi, name in PROB_BINS:
            sel = (g.max_prob >= lo) & (g.max_prob < hi) if lo > 0 else g.max_prob < hi
            row[name] = sel.sum() / n_folds
            row[f"{name} frac"] = sel.mean()
        rows.append(row)
    return pd.DataFrame(rows)


def missed_by_all(pl):
    miss = pl[~pl[[f"det_{m}" for m in MODELS]].any(axis=1)]
    cols = KEY + ["size_vox", "size_mm3", "nearest_lesion_mm"] + [f"max_prob_{m}" for m in MODELS]
    return miss[cols].sort_values(KEY)


def isolation(pl):
    rows = []
    small = pl[pl.size_vox <= SMALL_VOXELS]
    for dataset in DATASETS:
        d = small[small.dataset == dataset]
        for lo, hi, name in ISOLATION_BINS:
            g = d[(d.nearest_lesion_mm > lo) & (d.nearest_lesion_mm <= hi)]
            row = dict(dataset=dataset, nearest_lesion=name, n=len(g))
            for m in MODELS:
                row[f"recall_{m}"] = g[f"det_rate_{m}"].mean()
            rows.append(row)
    return pd.DataFrame(rows)


def clinical_size(pl, les):
    """Recall split at the MAGNIMS lesion size: "at least 3 mm in longest diameter" (Barkhof
    et al., Lancet Neurol 2025; Filippi et al., Lancet Neurol 2016), using the largest
    in-plane diameter from lesion_extent.csv. The 2025 consensus allows smaller lesions in
    some locations (fourth-ventricle floor, outer pons, corpus callosum), not modelled here."""
    pl = pl.assign(diameter_mm=pl.max_inplane_mm)
    rows = []
    for dataset in DATASETS:
        for name, sel in [("<3 mm", pl.diameter_mm < CLINICAL_DIAMETER_MM),
                          (">=3 mm", pl.diameter_mm >= CLINICAL_DIAMETER_MM),
                          (">=3 mm and <=150 vox", (pl.diameter_mm >= CLINICAL_DIAMETER_MM)
                           & (pl.size_vox <= SMALL_VOXELS))]:
            g = pl[(pl.dataset == dataset) & sel]
            row = dict(dataset=dataset, lesions=name, n=len(g), n_patients=g.case.map(patient_of).nunique())
            for m in MODELS:
                row[f"recall_{m}"] = g[f"det_rate_{m}"].mean()
            a, b, patients = fold_arrays(les, g, "CATMIL"), fold_arrays(les, g, "DiceCE"), \
                g.case.map(patient_of).tolist()
            w_scan, w_fold = boot_weights(patients, np.random.default_rng(SEED))
            boot = weighted_mean(a, w_scan, w_fold) - weighted_mean(b, w_scan, w_fold)
            row["delta_vs_DiceCE"] = a.mean() - b.mean()
            row["ci_low"], row["ci_high"] = np.nanpercentile(boot, [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


def fold_arrays(les, lesions, model):
    """Detection as array[fold, lesion] for the given lesions (rows of the per-lesion table)."""
    piv = les[les.model == model].pivot_table(index=KEY, columns="fold", values="detected")
    return piv.loc[pd.MultiIndex.from_frame(lesions[KEY])].to_numpy(float).T


def fold_stability(pl):
    rows = []
    for dataset in DATASETS:
        d = pl[(pl.dataset == dataset) & (pl.size_vox <= SMALL_VOXELS)]
        for m in MODELS:
            r = d[f"det_rate_{m}"]
            rows.append(dict(dataset=dataset, model=m, n_small=len(d), all_5=int((r == 1).sum()),
                             unstable_1_to_4=int(((r > 0) & (r < 1)).sum()), none=int((r == 0).sum())))
    return pd.DataFrame(rows)


def plot_recall_vs_size(pl, out_dir):
    edges = np.array([1, 3, 6, 11, 21, 51, 101, 151, 301, 1e6])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        d = pl[pl.dataset == dataset]
        idx = np.digitize(d.size_vox, edges) - 1
        centres = np.sqrt(edges[:-1] * np.minimum(edges[1:], 3000))
        for m in MODELS:
            colour, marker, ls = STYLE[m]
            r = [d.loc[idx == i, f"det_rate_{m}"].mean() for i in range(len(edges) - 1)]
            ax.plot(centres, r, ls=ls, marker=marker, color=colour, lw=1.6, ms=5, label=m,
                    zorder=3 if m == "CATMIL" else 2)
        counts = np.bincount(idx, minlength=len(edges) - 1)
        for c, n in zip(centres, counts):
            ax.text(c, -0.09, f"n={n}", ha="center", fontsize=7, color="#555")
        ax.set_xscale("log")
        ax.set_xlabel("Lesion size (voxels, log scale)")
        ax.set_title(dataset)
        ax.set_ylim(-0.13, 1.03)
        ax.grid(True, ls="--", lw=0.5, color="#c8c8c8")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Lesion recall")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ["pdf", "jpeg"]:
        fig.savefig(os.path.join(out_dir, f"recall_vs_size.{ext}"), dpi=300)


def plot_miss_proximity(mp, out_dir):
    shades = ["#d9d9d9", "#969696", "#e3a000"]
    hatches = ["", "//", ""]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, dataset in zip(axes, DATASETS):
        d = mp[mp.dataset == dataset].set_index("model").loc[MODELS]
        left = np.zeros(len(MODELS))
        for (_, _, name), c, h in zip(PROB_BINS, shades, hatches):
            ax.barh(MODELS, d[name], left=left, color=c, hatch=h, edgecolor="white", linewidth=2,
                    label=name)
            left += d[name].to_numpy()
        for y, v in enumerate(left):
            ax.text(v + left.max() * 0.01, y, f"{v:.0f}", va="center", fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Missed lesions per fold model (all test cases)")
        ax.set_title(dataset)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"max. probability: {l}" for l in labels], loc="lower center", ncol=3,
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ["pdf", "jpeg"]:
        fig.savefig(os.path.join(out_dir, f"miss_proximity.{ext}"), dpi=300)


def main(table_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10})
    les = pd.read_csv(f"{table_dir}/lesions.csv")
    extent = pd.read_csv(f"{table_dir}/lesion_extent.csv").drop(columns="size_vox")
    pl = per_lesion(les).merge(extent, on=KEY)
    pd.set_option("display.width", 200)
    for name, df in [("rescue_by_size", rescue_by_size(pl)), ("miss_proximity", miss_proximity(les)),
                     ("isolation", isolation(pl)), ("clinical_size", clinical_size(pl, les)),
                     ("fold_stability", fold_stability(pl)),
                     ("missed_by_all", missed_by_all(pl))]:
        df.to_csv(f"{out_dir}/{name}.csv", index=False)
        print(f"\n=== {name} ===")
        if name == "missed_by_all":
            print(df.groupby("dataset").agg(n=("size_vox", "size"), median_size=("size_vox", "median"),
                                            max_size=("size_vox", "max"),
                                            median_nearest_mm=("nearest_lesion_mm", "median"),
                                            median_max_prob_catmil=("max_prob_CATMIL", "median"),
                                            median_max_prob_dicece=("max_prob_DiceCE", "median")).round(3))
        else:
            print(df.round(3).to_string(index=False))
    plot_recall_vs_size(pl, out_dir)
    plot_miss_proximity(miss_proximity(les), out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:3])
