"""Combine the per-case lesion profiles into results/lesion_profile/MSLesSeg_lesion_profile.csv.

Reads results/lesion_profile/MSLesSeg/lesion_profile_<case>.csv (one row per 6-connected
ground-truth lesion, written by evaluation.utils.lesion_metrics_to_csv) and adds a two-group
KMeans cluster on standardized (sharpness, volume, contrast), as in notebooks/MSLesSeg.ipynb.
Profiles of predictions (file names with a model suffix, e.g. _UNETR) are skipped.

Usage: python analysis/combine_lesion_profile.py
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "results" / "lesion_profile" / "MSLesSeg"
OUT = REPO / "results" / "lesion_profile" / "MSLesSeg_lesion_profile.csv"
CASE = re.compile(r"lesion_profile_(P\d+_T\d+)\.csv$")


def main():
    dfs = []
    for path in sorted(SRC.glob("lesion_profile_*.csv")):
        m = CASE.match(path.name)
        if m is None:
            print("skipped", path.name)
            continue
        df = pd.read_csv(path)
        df["patient_id"] = m.group(1)
        df["id"] = df["patient_id"] + "_lesion_" + df["lesion_id"].astype(int).astype(str)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Rows with a NaN feature (e.g. contrast of a lesion with no valid ring) get no cluster.
    feats = ["sharpness", "volume_mm3", "contrast"]
    ok = df[feats].notna().all(axis=1)
    x = StandardScaler().fit_transform(df.loc[ok, feats].values)
    df.loc[ok, "cluster"] = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x).labels_
    df["cluster"] = df["cluster"].astype("Int64")

    df[["id", "patient_id", "lesion_id", "volume_mm3", "contrast", "sharpness", "cluster"]].to_csv(
        OUT, index=False)
    print(f"wrote {OUT}: {len(df)} lesions from {len(dfs)} scans")


if __name__ == "__main__":
    main()
