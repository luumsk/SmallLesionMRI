"""Export the result tables shared as Supplementary Materials of the journal paper.

Only derived numbers are exported (no images, masks or voxel coordinates), so the files can be
published under the article licence without redistributing the MSLesSeg or 3D-MR-MS data.

Outputs (in out_dir, default paper/journal/supplementary/):
  TableS1_test_per_case.csv     per-case metrics on the MSLesSeg and 3D-MR-MS test sets, for each
                                loss, post-processing setting and fold model (source of the loss
                                comparison, PP5 and ablation tables)
  TableS2_test_per_lesion.csv   one row per reference lesion, test case, loss and fold model:
                                size, longest diameters, detected or missed, coverage, max probability
                                (source of the lesion-size, lesion-fate and probability figures)
  TableS3_cv_per_scan.csv       per-scan metrics of the cross-validation predictions on the 81
                                MSLesSeg training scans and of the same fold models on the test set,
                                for each loss and architecture (source of the cross-validation tables)
  README.txt                    column definitions

Usage: python analysis/export_supplementary.py [out_dir]
"""
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "paper" / "journal" / "supplementary"
PER_FOLD = REPO / "results" / "per_fold"
LESIONS = REPO / "results" / "lesion_tables" / "lesions.csv"
EXTENT = REPO / "results" / "lesion_tables" / "lesion_extent.csv"
OOF = REPO / "results" / "analyses_2026-11" / "oof_per_scan.csv"

SMALL_VOX = 150  # small lesions: at most 150 voxels (evaluation/multiscale_eval.py)
DECIMALS = 6
LOSS_NAMES = {"nnUNet": "DiceCE", "CCDiceCE": "CC-DiceCE"}
# The seven metrics reported in the paper tables; the evaluation CSVs hold more columns that the
# paper does not use (and brier/ECE/entropy are not reported, see results/per_fold/README.md).
METRICS = ["small_lesion_recall", "fn_lesion_count", "fn_volume_fraction", "dice", "hd95_mm",
           "lesion_f1", "fp_volume_mm3"]
FILE_RE = re.compile(r"multiscale_eval_(?P<loss>[A-Za-z]+?)(?:_?pp(?P<pp>\d+))?_fold(?P<fold>\d)_(?:main|all)\.csv")


def reference_counts():
    """Number of reference lesions and small reference lesions per test case."""
    ext = pd.read_csv(EXTENT)
    ext["small"] = ext["size_vox"] <= SMALL_VOX
    return (ext.groupby(["dataset", "case"])
               .agg(n_reference_lesions=("lesion_id", "size"), n_small_reference_lesions=("small", "sum"))
               .reset_index().rename(columns={"case": "case_id"}))


def table_s1():
    rows = []
    for dataset, folder in [("MSLesSeg", PER_FOLD / "MSLesSeg" / "main"),
                            ("3D-MR-MS", PER_FOLD / "Ljubljana" / "all")]:
        for f in sorted(folder.glob("multiscale_eval_*_fold[0-4]_*.csv")):
            m = FILE_RE.fullmatch(f.name)
            if m is None:
                raise ValueError(f"unexpected file name {f}")
            df = pd.read_csv(f)[["case_id"] + METRICS]
            df.insert(0, "fold", int(m["fold"]))
            df.insert(0, "post_processing", f"PP{m['pp']}" if m["pp"] else "none")
            df.insert(0, "loss", LOSS_NAMES.get(m["loss"], m["loss"]))
            df.insert(0, "dataset", dataset)
            rows.append(df)
    s1 = pd.concat(rows, ignore_index=True).merge(reference_counts(), on=["dataset", "case_id"], how="left")
    if s1["n_reference_lesions"].isna().any():
        raise ValueError("test case missing from lesion_extent.csv")
    cols = ["dataset", "loss", "post_processing", "fold", "case_id",
            "n_reference_lesions", "n_small_reference_lesions"] + METRICS
    order = {"CATMIL": 0, "DiceCE": 1, "Tversky": 2, "FocalTversky": 3, "CC-DiceCE": 4, "CAT": 5, "MIL": 6}
    s1 = s1[cols].sort_values(
        ["dataset", "loss", "post_processing", "fold", "case_id"],
        key=lambda c: c.map(order) if c.name == "loss" else
                      c.map(lambda p: -1 if p == "none" else int(p[2:])) if c.name == "post_processing" else c)
    return s1.sort_values("dataset", key=lambda c: c.map({"MSLesSeg": 0, "3D-MR-MS": 1}), kind="stable")


def table_s2():
    les = pd.read_csv(LESIONS)
    ext = pd.read_csv(EXTENT)[["dataset", "case", "lesion_id", "max_inplane_mm", "feret_3d_mm"]]
    s2 = les.merge(ext, on=["dataset", "case", "lesion_id"], how="left", validate="many_to_one")
    if s2["max_inplane_mm"].isna().any():
        raise ValueError("lesion missing from lesion_extent.csv")
    s2 = s2.rename(columns={"model": "loss", "case": "case_id", "max_prob": "max_probability",
                            "max_inplane_mm": "longest_inplane_diameter_mm",
                            "feret_3d_mm": "longest_3d_diameter_mm"})
    s2["small"] = s2["size_vox"] <= SMALL_VOX
    return s2[["dataset", "loss", "fold", "case_id", "lesion_id", "size_vox", "size_mm3",
               "longest_inplane_diameter_mm", "longest_3d_diameter_mm", "small",
               "detected", "coverage", "max_probability"]]


def table_s3():
    oof = pd.read_csv(OOF)
    arch = oof["model"].str.extract(r"^(?P<a>.+?) \((?P<l>.+)\)$")
    oof.insert(0, "loss", arch["l"].fillna(oof["model"]))
    oof.insert(0, "architecture", arch["a"].fillna("nnU-Net"))
    oof = oof.rename(columns={"case": "case_id", "n_lesions": "n_reference_lesions",
                              "small_recall": "small_lesion_recall", "fn_count": "fn_lesion_count",
                              "small_recall_pp5": "small_lesion_recall_pp5",
                              "fn_count_pp5": "fn_lesion_count_pp5"})
    oof["split"] = oof["split"].map({"oof": "cross-validation", "test": "test"})
    return oof[["split", "architecture", "loss", "fold", "patient", "case_id", "n_reference_lesions",
                "dice", "small_lesion_recall", "fn_lesion_count", "lesion_f1",
                "small_lesion_recall_pp5", "fn_lesion_count_pp5", "lesion_f1_pp5"]]


README = f"""Supplementary Materials: per-case and per-lesion results
=======================================================

Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation
in Brain MRI. Code: https://github.com/luumsk/SmallLesionMRI (analysis/export_supplementary.py
writes these files from the evaluation outputs in results/).

The files contain derived metrics only; no images or masks. MSLesSeg case IDs are
<patient>_<timepoint> (for example P13_T1); 3D-MR-MS case IDs are the dataset's patient IDs.
Lesions are 6-connected components of the reference mask; a small lesion has at most
{SMALL_VOX} voxels in either dataset (MSLesSeg voxels are 1 mm^3; 3D-MR-MS voxels are about
0.18 mm^3, so size_mm3 differs from size_vox there). A reference lesion counts as detected if the
predicted mask (foreground probability > 0.5) overlaps it by at least one voxel.
Losses: CATMIL (ours), DiceCE, Tversky, FocalTversky, CC-DiceCE; CAT and MIL are the
ablations with one of the two terms. fold = the cross-validation fold model (0-4) that made
the prediction. Paper tables report the mean over cases within each fold, then the mean and
sample standard deviation over the five folds. Values are rounded to {DECIMALS} decimals.
Empty cells: metric undefined (for example small-lesion recall for a case without small lesions).

TableS1_test_per_case.csv: one row per dataset, loss, post-processing, fold and test case
  dataset                    MSLesSeg (12 test scans) or 3D-MR-MS (6 test scans, external)
  loss                       training loss (nnU-Net)
  post_processing            none, or PPs = predicted components smaller than s voxels removed
  fold, case_id              see above
  n_reference_lesions        reference lesions in the case
  n_small_reference_lesions  of which small
  small_lesion_recall        fraction of small reference lesions detected
  fn_lesion_count            reference lesions missed (all sizes)
  fn_volume_fraction         volume of missed reference lesions / total reference lesion volume
  dice                       voxel-wise Dice similarity coefficient
  hd95_mm                    95th-percentile Hausdorff distance (mm)
  lesion_f1                  lesion-wise F1 (harmonic mean of lesion-wise precision and recall)
  fp_volume_mm3              volume of predicted voxels outside the reference mask

TableS2_test_per_lesion.csv: one row per reference lesion, loss and fold, no post-processing
  dataset, loss, fold, case_id   as in Table S1
  lesion_id                      lesion number within the case's reference mask
  size_vox, size_mm3             lesion size
  longest_inplane_diameter_mm    longest diameter within any axial, coronal or sagittal slice
                                 (edge to edge; the measure behind the 3 mm clinical criterion)
  longest_3d_diameter_mm         longest 3D (Feret) diameter, edge to edge
  small                          size_vox <= {SMALL_VOX}
  detected                       True if detected (see above)
  coverage                       fraction of the lesion's voxels inside the predicted mask
  max_probability                highest predicted foreground probability inside the lesion

TableS3_cv_per_scan.csv: one row per split, model, fold and scan
  split                     cross-validation: each of the 81 MSLesSeg training scans predicted by
                            the fold model that did not train on it; test: the MSLesSeg test scans
                            predicted by each fold model
  architecture, loss        network and training loss
  fold, patient, case_id    fold model, patient ID, scan ID
  n_reference_lesions       reference lesions in the scan
  dice, small_lesion_recall, fn_lesion_count, lesion_f1   as in Table S1
  *_pp5                     the same metrics after removing predicted components under 5 voxels
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in [("TableS1_test_per_case", table_s1()), ("TableS2_test_per_lesion", table_s2()),
                     ("TableS3_cv_per_scan", table_s3())]:
        df = df.round(DECIMALS)
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(df)} rows, {df.shape[1]} columns")
    (OUT / "README.txt").write_text(README)
    print(f"written to {OUT}")


if __name__ == "__main__":
    main()
