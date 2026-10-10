"""Summarize the output of periventricular_flip.py.

For each dataset, loss, post-processing variant and distance D, averaged over
the five fold models:
  scans_gt_pos     scans whose reference has >= 1 periventricular lesion
  flip_to_neg      reference-positive scans predicted negative (prediction-based indicator)
  flip_to_pos      reference-negative scans predicted positive (false-positive components)
  pv_missed        periventricular reference lesions with no overlapping prediction
  pv_missed_frac   pv_missed / periventricular reference lesions
  pv_small_missed  same, for lesions <= 150 voxels

Usage: python periventricular_summary.py ../metrics/periventricular/pv_scans.csv
"""
import sys

import pandas as pd

df = pd.read_csv(sys.argv[1])
df["flip_to_neg"] = df.indicator_gt & ~df.indicator_pred
df["flip_to_pos"] = ~df.indicator_gt & df.indicator_pred
keys = ["dataset", "loss", "variant", "d_mm"]
per_fold = df.groupby(keys + ["fold"]).agg(
    scans=("case", "nunique"),
    scans_gt_pos=("indicator_gt", "sum"),
    flip_to_neg=("flip_to_neg", "sum"),
    flip_to_pos=("flip_to_pos", "sum"),
    pv_lesions=("gt_pv_lesions", "sum"),
    pv_missed=("pv_lesions_missed", "sum"),
    pv_small=("gt_pv_small_lesions", "sum"),
    pv_small_missed=("pv_small_missed", "sum"),
).reset_index()
summary = per_fold.groupby(keys).mean(numeric_only=True).drop(columns="fold")
summary["pv_missed_frac"] = summary.pv_missed / summary.pv_lesions
summary["pv_small_missed_frac"] = summary.pv_small_missed / summary.pv_small
pd.set_option("display.width", 220, "display.max_rows", 200, "display.multi_sparse", False)
print(summary.round(3).to_string())
