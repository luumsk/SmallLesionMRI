import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams

# Use Times font
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["axes.titleweight"] = "bold"
rcParams["axes.labelweight"] = "normal"

# Data
sizes = ["≤10 voxels", "≤50 voxels", "≤200 voxels", ">200 voxels"]

models = {
    "CATMIL":        [0.325924, 0.792450, 0.948488, 1.000000],
    "DiceCE":        [0.138272, 0.705382, 0.903844, 0.987498],
    "Tversky":       [0.200000, 0.747554, 0.932450, 0.995832],
    "FocalTversky":  [0.212346, 0.751288, 0.928144, 0.993750],
}

x = np.arange(len(sizes))
width = 0.2

fig, ax = plt.subplots(figsize=(9, 5.5))

# Plot grouped bars
for i, (model, values) in enumerate(models.items()):
    bars = ax.bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=model,
        edgecolor="black",
        linewidth=0.8,
    )

# Labels and formatting
ax.set_xticks(x)
ax.set_xticklabels(sizes)
ax.set_ylabel("Recall", fontsize=12)
ax.set_xlabel("Lesion Size", fontsize=12)

ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=10)

# Value labels
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

# Improve layout
plt.tight_layout()

# Save figure
plt.savefig("fn_recall_by_size.jpeg", dpi=720, format="jpeg")
