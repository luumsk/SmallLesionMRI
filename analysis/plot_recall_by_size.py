"""Lesion recall by lesion size for CATMIL and the baseline losses.

Recall per size bin is read from the per-fold summary JSON files (case-averaged
recall), then averaged over the five fold models. Bins follow multiscale_eval.py:
tiny <=10, small 11-50, medium 51-200, large >200 voxels.

Usage: python plot_recall_by_size.py <metrics_dir> <out_dir>
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DATASET = "MSLesSeg"
MODELS = {"CATMIL": "CATMIL", "DiceCE": "nnUNet", "Tversky": "Tversky", "FocalTversky": "FocalTversky"}
BINS = ["tiny", "small", "medium", "large"]
BIN_LABELS = ["≤10", "11–50", "51–200", ">200"]
HATCHES = {"CATMIL": "", "DiceCE": "//", "Tversky": "\\\\", "FocalTversky": "xx"}
CATMIL_COLOUR = "#e3a000"
BASE_COLOUR = "#a0a0a0"


def recall(metrics_dir, model):
    files = sorted(glob.glob(f"{metrics_dir}/{DATASET}/all/multiscale_eval_{model}_fold*_all.json"))
    assert len(files) == 5
    summaries = [json.load(open(f)) for f in files]
    return np.array([[s[f"recall_{b}"] for b in BINS] for s in summaries]).mean(0)


def main(metrics_dir, out_dir):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 11, "hatch.linewidth": 0.8})
    x = np.arange(len(BINS))
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.9))
    for i, (name, model) in enumerate(MODELS.items()):
        r = recall(metrics_dir, model)
        print(name, np.round(r, 2))
        bars = ax.bar(x + (i - 1.5) * width, r, width,
                      color=CATMIL_COLOUR if name == "CATMIL" else BASE_COLOUR,
                      edgecolor="black", linewidth=0.8, hatch=HATCHES[name], label=name)
        for bar, v in zip(bars, r):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Lesion size (voxels)")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1.08)
    ax.yaxis.grid(True, ls="--", lw=0.6, color="#c0c0c0")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    for ext in ["pdf", "jpeg"]:
        fig.savefig(os.path.join(out_dir, f"fn_recall_by_size.{ext}"), dpi=300)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
