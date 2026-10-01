"""Lesion-level figures for the loss comparison, from build_lesion_tables.py output.

  recall_vs_size_curve   lesion recall against lesion size on a log axis
  lesion_fate            each ground-truth lesion paired between CATMIL and a baseline
  max_prob_by_size       per-lesion maximum foreground probability, by size bin
  operating_curve        lesion recall against false-positive components per case
  lesion_gallery         FLAIR crops with contours and probability maps (MSLesSeg only)

Sizes are in mm^3 (equal to voxels on the 1 mm isotropic MSLesSeg data). Detection
is any-voxel overlap with 6-connected components, as in multiscale_eval.py.

Usage: python plot_lesion_figures.py <tables_dir> <out_dir>
"""
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = "/Volumes/BACH2TB/Tunka"
RAW = {"MSLesSeg": "nnUNet_raw/Dataset333_MSLesSeg", "3D-MR-MS": "nnUNet_raw/Dataset666_LjubljanaMS"}
PRED = {"MSLesSeg": "SmallLesionMRI/MSLesSeg/final_checkpoints",
        "3D-MR-MS": "SmallLesionMRI/LjubljanaMS/final_checkpoints"}
SUFFIX = {"MSLesSeg": "", "3D-MR-MS": "_3dmrms"}
MODEL_DIR = {"CATMIL": "CATMIL", "DiceCE": "nnUNet", "Tversky": "Tversky", "FocalTversky": "FocalTversky"}
MODELS = list(MODEL_DIR)
BASELINES = MODELS[1:]
STYLE = {
    "CATMIL": dict(color="#e3a000", ls="-", marker="o", lw=2.8, zorder=5),
    "DiceCE": dict(color="#0072b2", ls="--", marker="s", lw=1.5, zorder=3),
    "Tversky": dict(color="#009e73", ls="-.", marker="^", lw=1.5, zorder=3),
    "FocalTversky": dict(color="#cc79a7", ls=":", marker="D", lw=1.8, zorder=3),
}
BINS = [(0, 10, "≤10"), (10, 50, "11–50"), (50, 200, "51–200"), (200, np.inf, ">200")]
INK = "#3a3a3a"
GRID = dict(ls="--", lw=0.6, color="#c8c8c8")


def size_bin(mm3):
    for i, (lo, hi, _) in enumerate(BINS):
        if lo < mm3 <= hi or (i == 0 and mm3 <= hi):
            return i
    raise ValueError(mm3)


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)


def save(fig, out_dir, name):
    for ext in ["pdf", "jpeg"]:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- recall vs size

def plot_recall_curve(les, out_dir, suffix):
    edges = np.array([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, np.inf])
    ref = les[(les.model == "CATMIL") & (les.fold == 0)]
    counts = np.histogram(ref.size_mm3, edges)[0]
    keep = counts > 0
    centres = np.array([np.median(ref.size_mm3[(ref.size_mm3 > lo) & (ref.size_mm3 <= hi)])
                        if n else np.nan for lo, hi, n in zip(edges[:-1], edges[1:], counts)])

    fig, (ax, axn) = plt.subplots(2, 1, figsize=(8, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
    for model in MODELS:
        per_fold = []
        for _, g in les[les.model == model].groupby("fold"):
            det = np.histogram(g.size_mm3[g.detected], edges)[0]
            tot = np.histogram(g.size_mm3, edges)[0]
            per_fold.append(np.where(tot > 0, det / np.maximum(tot, 1), np.nan))
        per_fold = np.array(per_fold)
        st = STYLE[model]
        ax.fill_between(centres[keep], per_fold.min(0)[keep], per_fold.max(0)[keep],
                        color=st["color"], alpha=0.22 if model == "CATMIL" else 0.10, lw=0,
                        zorder=st["zorder"] - 2)
        ax.plot(centres[keep], per_fold.mean(0)[keep], color=st["color"], ls=st["ls"], lw=st["lw"],
                marker=st["marker"], ms=6.5 if model == "CATMIL" else 5, mec="white", mew=0.8,
                zorder=st["zorder"], label=model)
    for x in [10, 50, 200]:
        for a in (ax, axn):
            a.axvline(x, color="#9a9a9a", lw=0.7, ls=(0, (2, 3)), zorder=0)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Lesion recall")
    ax.yaxis.grid(True, **GRID)
    ax.legend(loc="lower right", frameon=False, handlelength=3)
    style_axes(ax)

    axn.bar(centres[keep], counts[keep], width=centres[keep] * 0.35, color="#bdbdbd", edgecolor="none")
    for c, n in zip(centres[keep], counts[keep]):
        axn.text(c, n, str(n), ha="center", va="bottom", fontsize=8, color=INK)
    axn.set_ylabel("Lesions")
    axn.set_ylim(0, counts.max() * 1.35)
    axn.set_xlabel("Lesion size (mm³, log scale)")
    style_axes(axn)
    save(fig, out_dir, f"recall_vs_size_curve{suffix}")


# ---------------------------------------------------------------- lesion fate

CATMIL_ONLY = "#e3a000"
BASE_ONLY = "#2a5caa"


def fate_table(les, baseline):
    """Mean lesions per fold model in each outcome, pairing CATMIL and the baseline fold by fold."""
    key = ["fold", "case", "lesion_id"]
    a = les[les.model == "CATMIL"].set_index(key)
    b = les[les.model == baseline].set_index(key)
    j = a[["size_mm3", "detected"]].join(b[["detected"]], rsuffix="_b")
    j["bin"] = j.size_mm3.map(size_bin)
    j["fate"] = np.select([j.detected & j.detected_b, j.detected & ~j.detected_b,
                           ~j.detected & j.detected_b], ["both", "catmil", "base"], "neither")
    n_folds = j.index.get_level_values("fold").nunique()
    return j.groupby(["bin", "fate"]).size().unstack(fill_value=0).reindex(
        index=range(len(BINS)), columns=["both", "catmil", "base", "neither"], fill_value=0) / n_folds


def plot_fate(les, out_dir, suffix):
    tables = {b: fate_table(les, b) for b in BASELINES}
    lim = max(max(t.catmil.max(), t.base.max()) for t in tables.values()) * 1.3
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    y = np.arange(len(BINS))
    for ax, (base, t) in zip(axes, tables.items()):
        ax.barh(y, t.catmil, height=0.55, color=CATMIL_ONLY, edgecolor="white", lw=1.5)
        ax.barh(y, -t.base, height=0.55, color=BASE_ONLY, edgecolor="white", lw=1.5)
        for yi in y:
            c, b_ = t.catmil[yi], t.base[yi]
            ax.text(c + lim * 0.02, yi, f"{c:.1f}", va="center", ha="left", fontsize=8, color=INK)
            ax.text(-b_ - lim * 0.02, yi, f"{b_:.1f}", va="center", ha="right", fontsize=8, color=INK)
            ax.text(lim * 0.98, yi + 0.36, f"both {t.both[yi]:.1f} · neither {t.neither[yi]:.1f}",
                    ha="right", va="center", fontsize=7, color="#6b6b6b")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_yticks(y)
        ax.set_yticklabels([b[2] for b in BINS])
        ax.invert_yaxis()
        ax.set_title(f"CATMIL vs {base}\n(total: CATMIL only {t.catmil.sum():.1f}, {base} only {t.base.sum():.1f})",
                     fontsize=9.5)
        ax.set_xlabel(f"← {base} only    Lesions per fold model    CATMIL only →", fontsize=8.5)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{abs(v):g}"))
        ax.xaxis.grid(True, **GRID)
        style_axes(ax)
    axes[0].set_ylabel("Lesion size (mm³)")
    fig.tight_layout()
    save(fig, out_dir, f"lesion_fate{suffix}")


# ---------------------------------------------------------------- max probability

PROB_LEVELS = [("absent", "p < 0.1", "#d9d9d9"), ("sub", "0.1 ≤ p < 0.5", "#7fa3d6"),
               ("above", "p ≥ 0.5 (detected)", "#2a5caa")]


def plot_max_prob(les, out_dir, suffix):
    # one value per lesion and fold model: the maximum foreground probability inside the lesion
    les = les.assign(bin=les.size_mm3.map(size_bin),
                     level=np.select([les.max_prob < 0.1, les.max_prob < 0.5], ["absent", "sub"], "above"))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.6), sharey=True, sharex=True)
    x = np.arange(len(MODELS))
    for b, ax in enumerate(axes.flat):
        sub = les[les.bin == b]
        frac = sub.groupby("model")["level"].value_counts(normalize=True).unstack(fill_value=0).reindex(
            index=MODELS, columns=[k for k, _, _ in PROB_LEVELS], fill_value=0)
        bottom = np.zeros(len(MODELS))
        for key, lab, colour in PROB_LEVELS:
            v = frac[key].values
            ax.bar(x, v, bottom=bottom, width=0.7, color=colour, edgecolor="white", lw=1.5,
                   label=lab if b == 0 else None)
            for xi, (bt, vi) in enumerate(zip(bottom, v)):
                if vi >= 0.07:
                    ax.text(xi, bt + vi / 2, f"{vi:.2f}", ha="center", va="center", fontsize=7.5,
                            color="white" if key == "above" else INK)
            bottom += v
        n = sub[(sub.model == "CATMIL") & (sub.fold == 0)].shape[0]
        ax.set_title(f"{BINS[b][2]} mm³ (n={n})", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=8.5)
        for tick, m in zip(ax.get_xticklabels(), MODELS):
            if m == "CATMIL":
                tick.set_fontweight("bold")
        ax.set_ylim(0, 1)
        style_axes(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("Fraction of lesions")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0),
               title="Maximum foreground probability inside the lesion", title_fontsize=9)
    save(fig, out_dir, f"max_prob_by_size{suffix}")


# ---------------------------------------------------------------- operating curve

MIN_SIZES = [0, 2, 3, 5, 8, 10, 15, 20]
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def operating_point(les_m, comp_m, t, s):
    n_cases = les_m.case.nunique()
    rec, fp = [], []
    for fold, g in les_m.groupby("fold"):
        rec.append((g[f"hit_size_t{t}"] >= max(s, 1)).mean())
        c = comp_m[(comp_m.fold == fold) & (comp_m.threshold == t)]
        fp.append(((c.size_vox >= s) & ~c.matched).sum() / n_cases)
    return np.mean(fp), np.mean(rec)


def plot_operating_curve(les, comp, out_dir, suffix):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for model in MODELS:
        st = STYLE[model]
        lm, cm = les[les.model == model], comp[comp.model == model]
        kw = dict(color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=4.5,
                  mec="white", mew=0.6, zorder=st["zorder"])
        pts = np.array([operating_point(lm, cm, 0.5, s) for s in MIN_SIZES])
        axes[0].plot(pts[:, 0], pts[:, 1], label=model, **kw)
        axes[0].plot(*pts[0], marker="o", ms=10, mfc="none", mec=st["color"], mew=1.4)
        axes[0].plot(*pts[MIN_SIZES.index(5)], marker="*", ms=13, color=st["color"], mec=INK, mew=0.6)
        pts = np.array([operating_point(lm, cm, t, 0) for t in THRESHOLDS])
        axes[1].plot(pts[:, 0], pts[:, 1], label=model, **kw)
        axes[1].plot(*pts[THRESHOLDS.index(0.5)], marker="o", ms=10, mfc="none", mec=st["color"], mew=1.4)
        if model == "CATMIL":
            for t, (x, y) in zip(THRESHOLDS, pts):
                if t in (0.1, 0.5, 0.9):
                    axes[1].annotate(f"t={t}", (x, y), xytext=(4, 4), textcoords="offset points",
                                     fontsize=7.5, color=INK)
            for s, (x, y) in zip(MIN_SIZES, np.array([operating_point(lm, cm, 0.5, s) for s in MIN_SIZES])):
                if s in (0, 5, 20):
                    axes[0].annotate(f"s={s}", (x, y), xytext=(4, 4), textcoords="offset points",
                                     fontsize=7.5, color=INK)
    axes[0].set_title("Minimum component size s (threshold 0.5)", fontsize=10)
    axes[1].set_title("Probability threshold t (no size filter)", fontsize=10)
    axes[0].set_ylabel("Lesion recall")
    for ax in axes:
        ax.set_xlabel("False-positive components per case")
        ax.grid(True, **GRID)
        style_axes(ax)
    extra = [Line2D([], [], ls="none", marker="o", ms=9, mfc="none", mec=INK, label="default (t=0.5, s=0)"),
             Line2D([], [], ls="none", marker="*", ms=12, color="#bdbdbd", mec=INK, label="PP5 (s=5)")]
    h, lab = axes[0].get_legend_handles_labels()
    axes[1].legend(h + extra, lab + [e.get_label() for e in extra], loc="lower right",
                   frameon=False, fontsize=8.5, handlelength=3)
    fig.tight_layout()
    save(fig, out_dir, f"operating_curve{suffix}")


# ---------------------------------------------------------------- gallery

GALLERY_FOLD = 0
HALF = 12  # crop half-width in voxels (1 mm on MSLesSeg)


def pick_gallery_lesions(les):
    """Stated rule: fold-0 lesions of at most 50 mm^3, grouped by CATMIL/DiceCE outcome;
    within each group take the lesion(s) closest to the group's median size, one per case."""
    key = ["case", "lesion_id"]
    f = les[les.fold == GALLERY_FOLD]
    a = f[f.model == "CATMIL"].set_index(key)
    b = f[f.model == "DiceCE"].set_index(key)
    j = a.join(b[["detected"]], rsuffix="_dice")
    j = j[j.size_mm3 <= 50].reset_index()
    groups = [("CATMIL only", j.detected & ~j.detected_dice, 2),
              ("DiceCE only", ~j.detected & j.detected_dice, 1),
              ("missed by both", ~j.detected & ~j.detected_dice, 1)]
    rows = []
    for name, mask, k in groups:
        g = j[mask].copy()
        if g.empty:
            continue
        g["dist"] = (g.size_mm3 - g.size_mm3.median()).abs()
        g = g.sort_values(["dist", "case", "lesion_id"]).drop_duplicates("case")
        rows += [(name, r) for _, r in g.head(k).iterrows()]
    return rows


def load_case(dataset, case):
    from scipy import ndimage
    gt = np.asarray(nib.load(f"{ROOT}/{RAW[dataset]}/labelsTs/{case}.nii.gz").dataobj) > 0.5
    flair = np.asarray(nib.load(f"{ROOT}/{RAW[dataset]}/imagesTs/{case}_0003.nii.gz").dataobj).astype(np.float32)
    lab = ndimage.label(gt, structure=ndimage.generate_binary_structure(3, 1))[0]
    probs = {}
    for model, d in MODEL_DIR.items():
        p = np.load(f"{ROOT}/{PRED[dataset]}/{d}/fold_{GALLERY_FOLD}/{case}.npz")["probabilities"][1]
        probs[model] = p.transpose(2, 1, 0)
    return flair, lab, probs


def crop(vol, cx, cy, z):
    x0, y0 = int(round(cx)) - HALF, int(round(cy)) - HALF
    sl = vol[max(x0, 0):x0 + 2 * HALF + 1, max(y0, 0):y0 + 2 * HALF + 1, z]
    return np.rot90(sl)


def plot_gallery(les, out_dir):
    dataset = "MSLesSeg"
    rows = pick_gallery_lesions(les[les.dataset == dataset])
    cols = ["FLAIR"] + MODELS
    fig, axes = plt.subplots(2 * len(rows), len(cols), figsize=(1.6 * len(cols), 1.6 * 2 * len(rows)),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.06})
    cache = {}
    for r, (group, row) in enumerate(rows):
        if row.case not in cache:
            cache = {row.case: load_case(dataset, row.case)}
        flair, lab, probs = cache[row.case]
        lesion = lab == row.lesion_id
        # axial slice through the lesion voxel with the highest probability summed over the four losses
        psum = sum(probs[m] for m in MODELS)
        z = int(np.unravel_index(np.argmax(np.where(lesion, psum, -1)), lesion.shape)[2])
        f = crop(flair, row.cx, row.cy, z)
        lo, hi = np.percentile(flair[flair > 0], [1, 99.5])
        g = crop(lesion, row.cx, row.cy, z)
        other = crop(lab > 0, row.cx, row.cy, z) & ~g
        for c, name in enumerate(cols):
            top, bot = axes[2 * r, c], axes[2 * r + 1, c]
            top.imshow(f, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
            if name == "FLAIR":
                bot.imshow(f, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
                for a in (top, bot):
                    a.contour(g, levels=[0.5], colors="#39d353", linewidths=1.1)
                    if other.any():
                        a.contour(other, levels=[0.5], colors="#39d353", linewidths=0.6, linestyles="dotted")
            else:
                p = crop(probs[name], row.cx, row.cy, z)
                top.contour(g, levels=[0.5], colors="#39d353", linewidths=1.1)
                if (p > 0.5).any():
                    top.contour(p > 0.5, levels=[0.5], colors="#ff5a1f", linewidths=1.1)
                im = bot.imshow(p, cmap="magma", vmin=0, vmax=1, interpolation="nearest")
                bot.contour(g, levels=[0.5], colors="#39d353", linewidths=1.1)
                mp = float(probs[name][lesion].max())
                bot.text(0.04, 0.04, f"max {mp:.2f}", transform=bot.transAxes, fontsize=7,
                         color="white", va="bottom")
            for a in (top, bot):
                a.set_xticks([])
                a.set_yticks([])
            if r == 0:
                top.set_title(name, fontsize=9)
        axes[2 * r, 0].set_ylabel(f"{row.case}\n{row.size_mm3:.0f} mm³\n{group}", fontsize=7.5)
        axes[2 * r + 1, 0].set_ylabel("probability", fontsize=7.5)
    cax = fig.add_axes([0.915, 0.3, 0.012, 0.4])
    fig.colorbar(im, cax=cax, label="Foreground probability")
    handles = [Line2D([], [], color="#39d353", lw=1.2, label="ground-truth lesion"),
               Line2D([], [], color="#39d353", lw=0.8, ls="dotted", label="other lesions"),
               Line2D([], [], color="#ff5a1f", lw=1.2, label="prediction (p > 0.5)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 0.08))
    save(fig, out_dir, "lesion_gallery")
    for group, row in rows:
        print(f"  gallery: {group:15s} {row.case} lesion {row.lesion_id} {row.size_mm3:.0f} mm3")


def main(tables_dir, out_dir):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 10, "hatch.linewidth": 0.6})
    les_all = pd.read_csv(f"{tables_dir}/lesions.csv")
    comp_all = pd.read_csv(f"{tables_dir}/components.csv")
    for dataset, suffix in SUFFIX.items():
        les = les_all[les_all.dataset == dataset]
        comp = comp_all[comp_all.dataset == dataset]
        if les.empty:
            continue
        print(dataset)
        plot_recall_curve(les, out_dir, suffix)
        plot_fate(les, out_dir, suffix)
        plot_max_prob(les, out_dir, suffix)
        plot_operating_curve(les, comp, out_dir, suffix)
    plot_gallery(les_all, out_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
