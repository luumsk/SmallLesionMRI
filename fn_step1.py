import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# -----------------------------
# Settings
# -----------------------------
csv_path = "all_metrics4.csv"
out_path = "fn_step1_combined.jpeg"

model_order = [
    "CATMIL",
    "FocalTversky",
    "Tversky",
    "nnUNet",
]

metric_info = [
    ("fn_lesion_count", "FN lesion count", "", False),
    ("lesion_recall", "Lesion recall", "", True),
    ("miss_rate", "Miss rate", "", True),
]

highlight_model = "CATMIL"

# -----------------------------
# Global style
# -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10


# -----------------------------
# Helpers
# -----------------------------
def load_plot_data(path: str) -> pd.DataFrame:
    """Load mean-row metrics and prepare model ordering."""
    df = pd.read_csv(path)
    df_mean = df[df["fold"].astype(str).str.lower() == "mean"].copy()

    name_map = {
        "DiceCE": "nnUNet",
    }
    df_mean["model_name"] = df_mean["model_name"].replace(name_map)

    df_plot = df_mean[df_mean["model_name"].isin(model_order)].copy()
    df_plot["model_name"] = pd.Categorical(
        df_plot["model_name"],
        categories=model_order,
        ordered=True,
    )
    df_plot = df_plot.sort_values("model_name").reset_index(drop=True)
    return df_plot


def add_value_labels(ax, bars, as_fraction: bool) -> None:
    """Add clean numeric labels above bars."""
    ymin, ymax = ax.get_ylim()
    y_offset = (ymax - ymin) * 0.025

    for bar in bars:
        height = bar.get_height()
        if as_fraction:
            label = f"{height:.3f}"
        else:
            label = f"{height:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )


def style_axis(ax, panel_label: str, y_label: str, as_fraction: bool) -> None:
    """Apply publication-style axis formatting."""
    ax.set_xlabel(panel_label)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(axis="x", rotation=0, length=0)
    ax.tick_params(axis="y", width=0.8)

    if as_fraction:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: f"{y:.2f}")
        )


# -----------------------------
# Load data
# -----------------------------
df_plot = load_plot_data(csv_path)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), dpi=300)

for ax, (metric, panel_label, y_label, as_fraction) in zip(axes, metric_info):
    values = df_plot[metric].values
    labels = df_plot["model_name"].tolist()
    x = list(range(len(labels)))

    bars = ax.bar(
        x,
        values,
        width=0.56,
        facecolor="#d9d9d9",
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

    for idx, bar in enumerate(bars):
        if labels[idx] == highlight_model:
            bar.set_facecolor("white")
            bar.set_hatch("////")
            bar.set_linewidth(1.2)

    ax.scatter(
        x,
        values,
        s=28,
        facecolors="white",
        edgecolors="black",
        linewidths=0.8,
        zorder=4,
    )

    catmil_idx = labels.index(highlight_model)
    ax.scatter(
        [catmil_idx],
        [values[catmil_idx]],
        s=34,
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(["CATMIL", "FocalTversky", "Tversky", "DiceCE"])

    if metric == "lesion_recall":
        ax.set_ylim(0.80, 0.93)
        ax.set_yticks(np.arange(0.80, 0.931, 0.03))
    elif metric == "miss_rate":
        ax.set_ylim(0.00, 0.17)
        ax.set_yticks(np.arange(0.00, 0.171, 0.04))
    else:
        ax.set_ylim(0.0, 5.5)
        ax.set_yticks(np.arange(0.0, 5.6, 1.0))

    ax.axvspan(-0.45, 0.45, color="#f2f2f2", zorder=0)

    style_axis(
        ax=ax,
        panel_label=panel_label,
        y_label=y_label,
        as_fraction=as_fraction,
    )
    add_value_labels(ax, bars, as_fraction=as_fraction)

fig.tight_layout(w_pad=1.8)
plt.savefig(out_path, format="jpeg", bbox_inches="tight", dpi=720)
