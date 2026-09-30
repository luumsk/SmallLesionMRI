"""Per-case paired plot of missed lesions (FN lesion count): CATMIL vs each baseline.

One panel per baseline (DiceCE, Tversky, FocalTversky). Each point is one test
case; coordinates are the mean FN count over the five fold models, error bars
span the fold min-max. Points below the diagonal are cases where CATMIL misses
fewer lesions than the baseline.

Usage: python plot_fn_per_case_paired.py <metrics_dir> <out_dir>
"""
import csv
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

METRIC = "fn_lesion_count"
DATASET = "MSLesSeg"
CATMIL = "CATMIL"
BASELINES = {"DiceCE": "nnUNet", "Tversky": "Tversky", "FocalTversky": "FocalTversky"}
POINT = "#2a5caa"
DIAG = "#6b6b6b"
SHADE = "#e8eef8"
INK = "#3a3a3a"


def load(metrics_dir, model):
    per_case = {}
    for f in sorted(glob.glob(f"{metrics_dir}/{DATASET}/all/multiscale_eval_{model}_fold*_all.csv")):
        for row in csv.DictReader(open(f)):
            per_case.setdefault(row["case_id"], []).append(float(row[METRIC]))
    return per_case


def paired(metrics_dir, baseline):
    c = load(metrics_dir, CATMIL)
    b = load(metrics_dir, baseline)
    cases = sorted(c)
    assert set(cases) == set(b) and all(len(c[k]) == len(b[k]) == 5 for k in cases)
    return cases, np.array([b[k] for k in cases]), np.array([c[k] for k in cases])


def panel(ax, cases, base, cat, name, lo, hi):
    bm, cm = base.mean(1), cat.mean(1)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=SHADE, zorder=0, lw=0)
    ax.plot([lo, hi], [lo, hi], color=DIAG, lw=1, ls="--", zorder=1)
    ax.errorbar(bm, cm,
                xerr=[bm - base.min(1), base.max(1) - bm],
                yerr=[cm - cat.min(1), cat.max(1) - cm],
                fmt="o", ms=5.5, color=POINT, ecolor=POINT, elinewidth=0.8,
                alpha=0.9, capsize=0, mec="white", mew=0.8, zorder=3)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Missed lesions per case, {name}")
    ax.set_title(f"CATMIL vs {name}", fontsize=11)
    ax.text(0.96, 0.05, "CATMIL misses fewer", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8.5, color=INK)
    ax.spines[["top", "right"]].set_visible(False)

    fewer, tie, more = int((cm < bm).sum()), int((cm == bm).sum()), int((cm > bm).sum())
    ax.text(0.04, 0.96,
            f"fewer {fewer}, tie {tie}, more {more}\n"
            f"mean {bm.mean():.2f} → {cm.mean():.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK)
    separated = [k for k, b, c in zip(cases, base, cat) if c.max() < b.min()]
    print(f"{name}: n={len(cases)} fewer={fewer} tie={tie} more={more} "
          f"{name}={bm.mean():.2f} CATMIL={cm.mean():.2f} separated={separated}")


def main(metrics_dir, out_dir):
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    data = {name: paired(metrics_dir, model) for name, model in BASELINES.items()}
    vmax = max(max(b.max(), c.max()) for _, b, c in data.values())
    lo, hi = -0.6, vmax * 1.08 + 0.5

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), sharey=True)
    for ax, (name, d) in zip(axes, data.items()):
        panel(ax, *d, name, lo, hi)
    axes[0].set_ylabel("Missed lesions per case, CATMIL")
    fig.tight_layout()
    for ext in ["pdf", "jpeg"]:
        fig.savefig(os.path.join(out_dir, f"fn_per_case_paired.{ext}"), dpi=300)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
