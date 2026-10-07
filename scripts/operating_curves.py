"""Detection/precision operating curves for CATMIL and the baseline losses.

Every loss is evaluated over a grid of probability thresholds t and minimum component
sizes s (post-processing), using the tables from build_lesion_tables.py. A lesion is
detected at (t, s) iff the largest predicted component at p > t that overlaps it has
>= s voxels; precision is the fraction of predicted components (>= s voxels) that
overlap the ground truth. Metrics follow multiscale_eval.py: computed per case, averaged
over cases, then over the five fold models. (t, s) = (0.5, 1) reproduces the paper table.

Outputs
  operating_grid.csv       recall / small-lesion recall / precision / F1 for every (t, s)
  iso_precision.csv        best recall of each loss at precision >= each baseline's
                           default precision (test-set selection, optimistic)
  transfer.csv             operating points chosen on MSLesSeg, applied unchanged to
                           3D-MR-MS, with patient-level bootstrap CIs and Wilcoxon p
  fp_components.csv        unmatched components per case by size, at t = 0.5
  operating_curves.{pdf,jpeg}

Usage: python operating_curves.py <lesion_table_dir> <out_dir>
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from stats_significance import SEED, boot_weights, patient_means, patient_of, weighted_mean

EPS = 1e-8
SMALL_VOXELS = 150
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MIN_SIZES = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 60, 85, 100, 170]
VOXEL_MM3 = {"MSLesSeg": 1.0, "3D-MR-MS": 0.1758}  # median voxel volume in lesions.csv
MODELS = ["CATMIL", "DiceCE", "Tversky", "FocalTversky"]
DATASETS = ["MSLesSeg", "3D-MR-MS"]
STYLE = {  # colour, marker, linestyle
    "CATMIL": ("#e3a000", "o", "-"),
    "DiceCE": ("#404040", "s", "--"),
    "Tversky": ("#808080", "^", "-."),
    "FocalTversky": ("#a8a8a8", "D", ":"),
}
FP_BINS = [(1, 1), (2, 4), (5, 9), (10, 49), (50, np.inf)]


def case_grid(les, comps):
    """Per (dataset, model, fold, case, t, s): recall, small recall, precision, F1."""
    keys = ["dataset", "model", "fold", "case"]
    cases = les[keys].drop_duplicates()
    rows = []
    for t in THRESHOLDS:
        hit = les[f"hit_size_t{t}"].to_numpy()
        c = comps[comps["threshold"] == t]
        for s in MIN_SIZES:
            det = les[keys].assign(det=hit >= s, small=les["size_vox"] <= SMALL_VOXELS)
            det["det_small"] = det["det"].astype(float).where(det["small"])
            g = det.groupby(keys).agg(recall=("det", "mean"), small_recall=("det_small", "mean"),
                                      n_gt=("det", "size"))
            k = c[c["size_vox"] >= s].groupby(keys).agg(n_pred=("matched", "size"),
                                                        tp_pred=("matched", "sum"))
            g = cases.merge(g.reset_index(), on=keys).merge(k.reset_index(), on=keys, how="left")
            g[["n_pred", "tp_pred"]] = g[["n_pred", "tp_pred"]].fillna(0)
            g["precision"] = g["tp_pred"] / (g["n_pred"] + EPS)
            g["f1"] = 2 * g["precision"] * g["recall"] / (g["precision"] + g["recall"] + EPS)
            g["t"], g["s"] = t, s
            rows.append(g)
    return pd.concat(rows, ignore_index=True)


def summarise(cg):
    metrics = ["recall", "small_recall", "precision", "f1", "n_pred"]
    per_fold = cg.groupby(["dataset", "model", "t", "s", "fold"])[metrics].mean()
    return per_fold.groupby(["dataset", "model", "t", "s"]).mean().reset_index()


def best_at_precision(grid, dataset, model, min_precision, target="small_recall"):
    g = grid[(grid.dataset == dataset) & (grid.model == model) & (grid.precision >= min_precision)]
    return None if g.empty else g.loc[g[target].idxmax()]


def iso_precision(grid):
    rows = []
    for dataset in DATASETS:
        for ref in MODELS[1:]:
            p0 = grid[(grid.dataset == dataset) & (grid.model == ref) & (grid.t == 0.5)
                      & (grid.s == 1)].precision.item()
            for model in MODELS:
                for target in ["small_recall", "recall"]:
                    b = best_at_precision(grid, dataset, model, p0, target)
                    rows.append(dict(dataset=dataset, precision_floor_from=ref, precision_floor=p0,
                                     model=model, target=target,
                                     t=None if b is None else b.t, s=None if b is None else b.s,
                                     value=np.nan if b is None else b[target],
                                     precision=np.nan if b is None else b.precision,
                                     f1=np.nan if b is None else b.f1))
    return pd.DataFrame(rows)


def arrays(cg, dataset, model, t, s, metric):
    """metric as array[fold, case] and the matching patient ids."""
    g = cg[(cg.dataset == dataset) & (cg.model == model) & (cg.t == t) & (cg.s == s)]
    piv = g.pivot(index="case", columns="fold", values=metric).sort_index()
    return piv.to_numpy(float).T, [patient_of(c) for c in piv.index]


def compare(cg, dataset, a_point, b_point, metric):
    """CATMIL at a_point vs reference at b_point: Δ, patient/fold bootstrap CI, Wilcoxon p."""
    a, patients = arrays(cg, dataset, "CATMIL", *a_point, metric)
    b, patients_b = arrays(cg, dataset, b_point[0], *b_point[1:], metric)
    assert patients == patients_b
    w_scan, w_fold = boot_weights(patients, np.random.default_rng(SEED))
    boot = weighted_mean(a, w_scan, w_fold) - weighted_mean(b, w_scan, w_fold)
    d = (patient_means(a, patients) - patient_means(b, patients)).dropna()
    p = wilcoxon(d, method="exact").pvalue if (d != 0).any() else 1.0
    delta = np.nanmean(np.nanmean(a, 0)) - np.nanmean(np.nanmean(b, 0))
    return delta, *np.nanpercentile(boot, [2.5, 97.5]), p


def to_target_voxels(s, src, dst):
    """Minimum size s (voxels on src) as the nearest grid size with the same volume on dst.
    s = 1 means no post-processing and stays 1."""
    if s == 1:
        return 1
    v = s * VOXEL_MM3[src] / VOXEL_MM3[dst]
    return min(MIN_SIZES, key=lambda m: abs(m - v))


def transfer(grid, cg):
    """Pick (t, s) on MSLesSeg, apply to 3D-MR-MS with s in the same voxels or the same mm³.

    Rule A (iso-precision): CATMIL's (t, s) with the highest small-lesion recall whose
      precision is >= the reference's default precision on MSLesSeg; reference at default.
    Rule B (best F1): every loss at its own F1-maximising (t, s) on MSLesSeg.
    """
    rows = []
    src, dst = "MSLesSeg", "3D-MR-MS"
    for ref in MODELS[1:]:
        p0 = grid[(grid.dataset == src) & (grid.model == ref) & (grid.t == 0.5)
                  & (grid.s == 1)].precision.item()
        a = best_at_precision(grid, src, "CATMIL", p0)
        rules = {"iso-precision": ((a.t, a.s), (ref, 0.5, 1))}
        fa = grid[(grid.dataset == src) & (grid.model == "CATMIL")].sort_values("f1").iloc[-1]
        fb = grid[(grid.dataset == src) & (grid.model == ref)].sort_values("f1").iloc[-1]
        rules["best-F1"] = ((fa.t, fa.s), (ref, fb.t, fb.s))
        for rule, (pa0, pb0) in rules.items():
            for dataset, units in [(src, "voxels"), (dst, "voxels"), (dst, "mm3")]:
                pa, pb = pa0, pb0
                if units == "mm3":  # same physical minimum size on the target dataset
                    pa = (pa[0], to_target_voxels(pa[1], src, dst))
                    pb = (pb[0], pb[1], to_target_voxels(pb[2], src, dst))
                row = dict(rule=rule, reference=ref, evaluated_on=dataset, min_size_units=units,
                           catmil_t=pa[0], catmil_s=pa[1], ref_t=pb[1], ref_s=pb[2])
                for metric in ["small_recall", "recall", "precision", "f1"]:
                    va = grid[(grid.dataset == dataset) & (grid.model == "CATMIL") & (grid.t == pa[0])
                              & (grid.s == pa[1])][metric].item()
                    vb = grid[(grid.dataset == dataset) & (grid.model == ref) & (grid.t == pb[1])
                              & (grid.s == pb[2])][metric].item()
                    d, lo, hi, p = compare(cg, dataset, pa, pb, metric)
                    row.update({f"{metric}_catmil": va, f"{metric}_ref": vb, f"{metric}_delta": d,
                                f"{metric}_ci_low": lo, f"{metric}_ci_high": hi, f"{metric}_p": p})
                rows.append(row)
    return pd.DataFrame(rows)


def fp_components(les, comps):
    keys = ["dataset", "model", "fold", "case"]
    n_cases = les[keys].drop_duplicates().groupby(["dataset", "model"]).size()
    c = comps[(comps.threshold == 0.5) & ~comps.matched]
    rows = []
    for (dataset, model), g in c.groupby(["dataset", "model"]):
        row = dict(dataset=dataset, model=model)
        for lo, hi in FP_BINS:
            name = f"{lo}" if lo == hi else (f">={lo}" if np.isinf(hi) else f"{lo}-{hi}")
            row[f"fp_cc_{name}"] = ((g.size_vox >= lo) & (g.size_vox <= hi)).sum() / n_cases[dataset, model]
        row["fp_cc_total"] = len(g) / n_cases[dataset, model]
        rows.append(row)
    return pd.DataFrame(rows)


def pareto(points):
    """Upper-right frontier of (precision, recall) points, sorted by precision."""
    pts = points.sort_values("precision", ascending=False)
    front, best = [], -1
    for _, r in pts.iterrows():
        if r.y > best:
            front.append(r)
            best = r.y
    return pd.DataFrame(front).sort_values("precision")


def plot(grid, out_dir):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(9, 7.6))
    for col, dataset in enumerate(DATASETS):
        for row, (target, label) in enumerate([("small_recall", f"Small-lesion recall (≤{SMALL_VOXELS} vox)"),
                                               ("recall", "Lesion recall")]):
            ax = axes[row, col]
            for model in MODELS:
                colour, marker, ls = STYLE[model]
                g = grid[(grid.dataset == dataset) & (grid.model == model)].assign(y=lambda d: d[target])
                ax.scatter(g.precision, g.y, s=6, color=colour, alpha=0.25, linewidths=0)
                f = pareto(g)
                ax.plot(f.precision, f.y, ls=ls, lw=1.6, color=colour, label=model,
                        zorder=3 if model == "CATMIL" else 2)
                d = g[(g.t == 0.5) & (g.s == 1)]
                ax.scatter(d.precision, d.y, s=60, marker=marker, color=colour, edgecolor="black",
                           linewidths=0.8, zorder=4)
                if model in ("CATMIL", "DiceCE"):
                    p5 = g[(g.t == 0.5) & (g.s == 5)]
                    ax.scatter(p5.precision, p5.y, s=60, marker=marker, facecolor="white",
                               edgecolor=colour, linewidths=1.4, zorder=4)
            ax.set_title(dataset if row == 0 else "")
            ax.set_xlabel("Lesion precision")
            ax.set_ylabel(label)
            ax.grid(True, ls="--", lw=0.5, color="#c8c8c8")
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles += [plt.Line2D([], [], marker="o", ls="", color="#777", markeredgecolor="black", markersize=7),
                plt.Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor="#777",
                           markersize=7)]
    labels += ["default (t=0.5)", "min. size 5 vox"]
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    for ext in ["pdf", "jpeg"]:
        fig.savefig(os.path.join(out_dir, f"operating_curves.{ext}"), dpi=300)


def main(table_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    les = pd.read_csv(f"{table_dir}/lesions.csv")
    comps = pd.read_csv(f"{table_dir}/components.csv")
    cg = case_grid(les, comps)
    grid = summarise(cg)
    grid.to_csv(f"{out_dir}/operating_grid.csv", index=False)

    default = grid[(grid.t == 0.5) & (grid.s == 1)]
    print("Default operating point (should match the paper table):")
    print(default[["dataset", "model", "small_recall", "recall", "precision", "f1"]].round(4).to_string(index=False))

    iso = iso_precision(grid)
    iso.to_csv(f"{out_dir}/iso_precision.csv", index=False)
    print("\nBest recall per loss at precision >= the reference's default precision (test-set selection):")
    print(iso.round(4).to_string(index=False))

    tr = transfer(grid, cg)
    tr.to_csv(f"{out_dir}/transfer.csv", index=False)
    print("\nOperating points chosen on MSLesSeg, applied to both datasets (Δ = CATMIL − reference):")
    for _, r in tr.iterrows():
        print(f"{r.rule:<13} vs {r.reference:<12} on {r.evaluated_on:<9} [{r.min_size_units}] "
              f"CATMIL(t={r.catmil_t},s={r.catmil_s:g}) "
              f"ref(t={r.ref_t},s={r.ref_s:g})")
        for m in ["small_recall", "recall", "precision", "f1"]:
            print(f"    {m:<13} {r[f'{m}_catmil']:.3f} vs {r[f'{m}_ref']:.3f}  Δ={r[f'{m}_delta']:+.3f} "
                  f"[{r[f'{m}_ci_low']:+.3f}, {r[f'{m}_ci_high']:+.3f}] p={r[f'{m}_p']:.3f}")

    fp = fp_components(les, comps)
    fp.to_csv(f"{out_dir}/fp_components.csv", index=False)
    print("\nUnmatched (false-positive) components per case at t=0.5, by size in voxels:")
    print(fp.round(2).to_string(index=False))

    plot(grid, out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:3])
