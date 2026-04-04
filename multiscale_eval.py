"""Multiscale evaluation for small-lesion segmentation.

This script matches ground-truth masks (*.nii.gz) with model predictions
(predicted mask *.nii.gz and probability *.npz), computes per-case metrics,
saves per-case results to a CSV, and saves mean metrics to a flat JSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage



EPS = 1e-8


@dataclass(frozen=True)
class CasePaths:
    """Paths for a single case."""

    case_id: str
    gt_path: Path
    pred_mask_path: Path
    pred_prob_path: Optional[Path]


def _stem_niigz(path: Path) -> str:
    """Return filename stem for .nii.gz (case id)."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")]
    return path.stem


def _list_niigz(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.nii.gz") if p.is_file()])


def _list_npz(folder: Path) -> List[Path]:
    return sorted([p for p in folder.glob("*.npz") if p.is_file()])


def build_case_table(
    gt_dir: Path,
    pred_mask_dir: Path,
    pred_prob_dir: Optional[Path],
    allow_missing_prob: bool,
) -> List[CasePaths]:
    """Match GT masks with predicted masks and optional probability files."""
    gt_paths = _list_niigz(gt_dir)
    pred_paths = _list_niigz(pred_mask_dir)

    gt_map = {_stem_niigz(p): p for p in gt_paths}
    pred_map = {_stem_niigz(p): p for p in pred_paths}

    prob_map: Dict[str, Path] = {}
    if pred_prob_dir is not None:
        for p in _list_npz(pred_prob_dir):
            prob_map[p.stem] = p

    shared = sorted(set(gt_map.keys()) & set(pred_map.keys()))
    if not shared:
        raise ValueError(
            "No matching case ids found between GT and prediction masks. "
            "Check filenames and extensions (.nii.gz)."
        )

    cases: List[CasePaths] = []
    for case_id in shared:
        prob_path = prob_map.get(case_id)
        if prob_path is None and not allow_missing_prob:
            raise FileNotFoundError(
                f"Missing probability .npz for case '{case_id}'."
            )
        cases.append(
            CasePaths(
                case_id=case_id,
                gt_path=gt_map[case_id],
                pred_mask_path=pred_map[case_id],
                pred_prob_path=prob_path,
            )
        )

    return cases


def load_nii_mask(path: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load NIfTI as a boolean mask and return voxel spacing (mm)."""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    mask = data.astype(np.float32) > 0.5

    hdr = img.header
    zooms = hdr.get_zooms()
    if len(zooms) < 3:
        raise ValueError(f"Invalid zooms for 3D image: {path}")

    spacing = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return mask, spacing


def load_npz_probability(path: Path) -> np.ndarray:
    """Load foreground probability volume from a .npz file."""
    obj = np.load(str(path))
    keys = list(obj.keys())
    if not keys:
        raise ValueError(f"Empty npz file: {path}")

    if "probabilities" in obj:
        arr = obj["probabilities"]
    elif "softmax" in obj:
        arr = obj["softmax"]
    else:
        arr = obj[keys[0]]

    arr = np.asarray(arr)

    if arr.ndim == 3:
        fg = arr
    elif arr.ndim == 4:
        if arr.shape[0] == 1:
            fg = arr[0]
        elif arr.shape[0] >= 2:
            fg = arr[1]
        else:
            raise ValueError(f"Unexpected probability shape: {arr.shape}")
    else:
        raise ValueError(f"Unexpected probability array ndim: {arr.ndim}")

    fg = np.clip(fg.astype(np.float32), 0.0, 1.0)
    return fg


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    """Dice for binary masks."""
    gt_sum = float(gt.sum())
    pr_sum = float(pred.sum())
    if gt_sum + pr_sum == 0.0:
        return 1.0
    inter = float(np.logical_and(gt, pred).sum())
    return (2.0 * inter) / (gt_sum + pr_sum + EPS)


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    """Return a boolean mask of surface voxels for a binary object."""
    if mask.sum() == 0:
        return np.zeros_like(mask, dtype=bool)

    struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
    eroded = ndimage.binary_erosion(mask, structure=struct, iterations=1)
    surface = np.logical_and(mask, np.logical_not(eroded))
    return surface


def _surface_distances_mm(
    src_surface: np.ndarray,
    dst_mask: np.ndarray,
    spacing: Tuple[float, float, float],
) -> np.ndarray:
    """Distances (mm) from each src surface voxel to nearest dst surface."""
    if src_surface.sum() == 0 or dst_mask.sum() == 0:
        return np.array([], dtype=np.float32)

    dst_surface = _surface_voxels(dst_mask)
    if dst_surface.sum() == 0:
        return np.array([], dtype=np.float32)

    dt = ndimage.distance_transform_edt(
        np.logical_not(dst_surface), sampling=spacing
    )
    dists = dt[src_surface]
    return dists.astype(np.float32)


def hd95_mm(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Tuple[float, float, float],
) -> float:
    """HD95 in mm using symmetric surface distances."""
    if gt.sum() == 0 and pred.sum() == 0:
        return 0.0
    if gt.sum() == 0 or pred.sum() == 0:
        return float("inf")

    s_gt = _surface_voxels(gt)
    s_pr = _surface_voxels(pred)
    d1 = _surface_distances_mm(s_gt, pred, spacing)
    d2 = _surface_distances_mm(s_pr, gt, spacing)
    if d1.size == 0 or d2.size == 0:
        return float("inf")

    all_d = np.concatenate([d1, d2], axis=0)
    return float(np.percentile(all_d, 95))


def assd_mm(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Tuple[float, float, float],
) -> float:
    """ASSD in mm using symmetric surface distances."""
    if gt.sum() == 0 and pred.sum() == 0:
        return 0.0
    if gt.sum() == 0 or pred.sum() == 0:
        return float("inf")

    s_gt = _surface_voxels(gt)
    s_pr = _surface_voxels(pred)
    d1 = _surface_distances_mm(s_gt, pred, spacing)
    d2 = _surface_distances_mm(s_pr, gt, spacing)
    if d1.size == 0 or d2.size == 0:
        return float("inf")

    return float((d1.mean() + d2.mean()) / 2.0)


def _label_cc(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
    lab, n = ndimage.label(mask, structure=struct)
    return lab.astype(np.int32), int(n)


def lesion_detection_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Tuple[float, float, float],
    small_voxels_thresh: int,
) -> Dict[str, float]:
    """Lesion-wise metrics using connected components."""
    gt_lab, gt_n = _label_cc(gt)
    pr_lab, pr_n = _label_cc(pred)

    gt_sizes = np.bincount(gt_lab.ravel())
    gt_ids = list(range(1, gt_n + 1))

    gt_hit = set()
    pr_hit = set()

    if gt_n > 0 and pr_n > 0:
        overlap_pairs = np.unique(
            np.stack([gt_lab[gt], pr_lab[gt]], axis=1), axis=0
        )
        for gt_id, pr_id in overlap_pairs:
            if gt_id == 0 or pr_id == 0:
                continue
            gt_hit.add(int(gt_id))
            pr_hit.add(int(pr_id))

    tp_gt = len(gt_hit)
    tp_pr = len(pr_hit)
    fn_gt = gt_n - tp_gt

    recall = tp_gt / (gt_n + EPS)
    precision = tp_pr / (pr_n + EPS)
    f1 = (2.0 * precision * recall) / (precision + recall + EPS)
    miss_rate = fn_gt / (gt_n + EPS)

    small_gt_ids = [
        i for i in gt_ids if int(gt_sizes[i]) <= int(small_voxels_thresh)
    ]
    if len(small_gt_ids) == 0:
        small_recall = float("nan")
    else:
        small_hit = sum([1 for i in small_gt_ids if i in gt_hit])
        small_recall = small_hit / (len(small_gt_ids) + EPS)

    missed_gt_ids = [i for i in gt_ids if i not in gt_hit]
    missed_fn_voxels = float(sum(int(gt_sizes[i]) for i in missed_gt_ids))
    voxel_volume_mm3 = _voxel_volume_mm3(spacing)
    total_gt_voxels = float(int(gt.sum()))
    fn_volume_mm3 = missed_fn_voxels * voxel_volume_mm3
    total_gt_volume_mm3 = total_gt_voxels * voxel_volume_mm3
    fn_volume_fraction = (
        fn_volume_mm3 / (total_gt_volume_mm3 + EPS)
        if total_gt_volume_mm3 > 0.0
        else float("nan")
    )

    return {
        "gt_lesion_count": float(gt_n),
        "pred_lesion_count": float(pr_n),
        "fn_lesion_count": float(fn_gt),
        "lesion_precision": float(precision),
        "lesion_recall": float(recall),
        "lesion_f1": float(f1),
        "miss_rate": float(miss_rate),
        "small_lesion_recall": float(small_recall),
        "fn_volume_mm3": float(fn_volume_mm3),
        "fn_volume_fraction": float(fn_volume_fraction),
    }


def _voxel_volume_mm3(spacing: Tuple[float, float, float]) -> float:
    return float(spacing[0] * spacing[1] * spacing[2])


def _nan_percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, q))


# -------- FN error profile helper functions and main function --------

def _safe_fraction(numerator: float, denominator: float) -> float:
    """Return a stable fraction with NaN when denominator is zero."""
    if denominator <= 0.0:
        return float("nan")
    return float(numerator) / float(denominator + EPS)


def _coverage_histogram_metrics(
    coverages: np.ndarray,
) -> Dict[str, float]:
    """Coverage histogram fractions for later plotting and analysis."""
    out: Dict[str, float] = {
        "lesion_coverage_hist_0_0_1_frac": float("nan"),
        "lesion_coverage_hist_0_1_0_25_frac": float("nan"),
        "lesion_coverage_hist_0_25_0_5_frac": float("nan"),
        "lesion_coverage_hist_0_5_0_75_frac": float("nan"),
        "lesion_coverage_hist_0_75_1_0_frac": float("nan"),
    }

    if coverages.size == 0:
        return out

    out["lesion_coverage_hist_0_0_1_frac"] = float(
        np.mean((coverages >= 0.0) & (coverages < 0.1))
    )
    out["lesion_coverage_hist_0_1_0_25_frac"] = float(
        np.mean((coverages >= 0.1) & (coverages < 0.25))
    )
    out["lesion_coverage_hist_0_25_0_5_frac"] = float(
        np.mean((coverages >= 0.25) & (coverages < 0.5))
    )
    out["lesion_coverage_hist_0_5_0_75_frac"] = float(
        np.mean((coverages >= 0.5) & (coverages < 0.75))
    )
    out["lesion_coverage_hist_0_75_1_0_frac"] = float(
        np.mean((coverages >= 0.75) & (coverages <= 1.0))
    )
    return out


def fn_error_profile(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Tuple[float, float, float],
) -> Dict[str, float]:
    """Characterize FN errors by lesion size and lesion coverage."""
    gt_lab, gt_n = _label_cc(gt)
    voxel_volume_mm3 = _voxel_volume_mm3(spacing)

    out: Dict[str, float] = {
        "fn_tiny_count": 0.0,
        "fn_small_count": 0.0,
        "fn_medium_count": 0.0,
        "fn_large_count": 0.0,
        "recall_tiny": float("nan"),
        "recall_small": float("nan"),
        "recall_medium": float("nan"),
        "recall_large": float("nan"),
        "fn_tiny_volume_mm3": 0.0,
        "fn_small_volume_mm3": 0.0,
        "fn_medium_volume_mm3": 0.0,
        "fn_large_volume_mm3": 0.0,
        "fn_tiny_volume_frac": float("nan"),
        "fn_small_volume_frac": float("nan"),
        "fn_medium_volume_frac": float("nan"),
        "fn_large_volume_frac": float("nan"),
        "lesion_coverage_mean": float("nan"),
        "lesion_coverage_median": float("nan"),
        "lesion_coverage_p25": float("nan"),
        "lesion_coverage_p75": float("nan"),
        "lesion_coverage_lt_0_1_frac": float("nan"),
        "lesion_coverage_lt_0_25_frac": float("nan"),
        "lesion_coverage_lt_0_5_frac": float("nan"),
        "lesion_coverage_zero_frac": float("nan"),
        "lesion_coverage_hist_0_0_1_frac": float("nan"),
        "lesion_coverage_hist_0_1_0_25_frac": float("nan"),
        "lesion_coverage_hist_0_25_0_5_frac": float("nan"),
        "lesion_coverage_hist_0_5_0_75_frac": float("nan"),
        "lesion_coverage_hist_0_75_1_0_frac": float("nan"),
    }

    if gt_n == 0:
        return out

    gt_sizes = np.bincount(gt_lab.ravel())
    gt_ids = list(range(1, gt_n + 1))

    coverages: List[float] = []

    bins = {
        "tiny": {"count": 0, "hit": 0, "fn": 0, "vol": 0.0, "fn_vol": 0.0},
        "small": {"count": 0, "hit": 0, "fn": 0, "vol": 0.0, "fn_vol": 0.0},
        "medium": {"count": 0, "hit": 0, "fn": 0, "vol": 0.0, "fn_vol": 0.0},
        "large": {"count": 0, "hit": 0, "fn": 0, "vol": 0.0, "fn_vol": 0.0},
    }

    for lesion_id in gt_ids:
        lesion_mask = gt_lab == lesion_id
        lesion_voxels = int(gt_sizes[lesion_id])
        lesion_volume_mm3 = float(lesion_voxels) * voxel_volume_mm3

        if lesion_voxels <= 10:
            bin_name = "tiny"
        elif lesion_voxels <= 50:
            bin_name = "small"
        elif lesion_voxels <= 200:
            bin_name = "medium"
        else:
            bin_name = "large"

        bins[bin_name]["count"] += 1
        bins[bin_name]["vol"] += lesion_volume_mm3

        overlap_voxels = float(np.logical_and(lesion_mask, pred).sum())
        coverage = overlap_voxels / (float(lesion_voxels) + EPS)
        coverages.append(float(coverage))

        if overlap_voxels > 0.0:
            bins[bin_name]["hit"] += 1
        else:
            bins[bin_name]["fn"] += 1
            bins[bin_name]["fn_vol"] += lesion_volume_mm3

    coverage_arr = np.asarray(coverages, dtype=np.float32)
    out["lesion_coverage_mean"] = float(np.mean(coverage_arr))
    out["lesion_coverage_median"] = float(np.median(coverage_arr))
    out["lesion_coverage_p25"] = _nan_percentile(coverage_arr, 25.0)
    out["lesion_coverage_p75"] = _nan_percentile(coverage_arr, 75.0)
    out["lesion_coverage_lt_0_1_frac"] = float(np.mean(coverage_arr < 0.1))
    out["lesion_coverage_lt_0_25_frac"] = float(np.mean(coverage_arr < 0.25))
    out["lesion_coverage_lt_0_5_frac"] = float(np.mean(coverage_arr < 0.5))
    out["lesion_coverage_zero_frac"] = float(np.mean(coverage_arr <= EPS))
    out.update(_coverage_histogram_metrics(coverage_arr))

    for bin_name in ["tiny", "small", "medium", "large"]:
        count_value = float(bins[bin_name]["count"])
        hit_value = float(bins[bin_name]["hit"])
        fn_value = float(bins[bin_name]["fn"])
        vol_value = float(bins[bin_name]["vol"])
        fn_vol_value = float(bins[bin_name]["fn_vol"])

        out[f"fn_{bin_name}_count"] = fn_value
        out[f"recall_{bin_name}"] = _safe_fraction(hit_value, count_value)
        out[f"fn_{bin_name}_volume_mm3"] = fn_vol_value
        out[f"fn_{bin_name}_volume_frac"] = _safe_fraction(
            fn_vol_value,
            vol_value,
        )

    return out


def fp_error_profile(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Tuple[float, float, float],
    near_mm: float,
    far_mm: float,
) -> Dict[str, float]:
    """Characterize FP errors as boundary extensions vs detached blobs."""
    fp = np.logical_and(pred, np.logical_not(gt))
    fp_vox = int(fp.sum())

    out: Dict[str, float] = {
        "fp_voxels": float(fp_vox),
        "fp_volume_mm3": float(fp_vox) * _voxel_volume_mm3(spacing),
        "fp_near_voxels_frac": float("nan"),
        "fp_far_voxels_frac": float("nan"),
        "fp_dist_median_mm": float("nan"),
        "fp_dist_p95_mm": float("nan"),
        "fp_boundary_cc": float("nan"),
        "fp_blob_cc": float("nan"),
        "fp_boundary_volume_mm3": float("nan"),
        "fp_blob_volume_mm3": float("nan"),
        "fp_blob_cc_frac": float("nan"),
        "fp_blob_volume_frac": float("nan"),
    }

    if fp_vox == 0:
        out.update(
            {
                "fp_near_voxels_frac": 0.0,
                "fp_far_voxels_frac": 0.0,
                "fp_dist_median_mm": 0.0,
                "fp_dist_p95_mm": 0.0,
                "fp_boundary_cc": 0.0,
                "fp_blob_cc": 0.0,
                "fp_boundary_volume_mm3": 0.0,
                "fp_blob_volume_mm3": 0.0,
                "fp_blob_cc_frac": 0.0,
                "fp_blob_volume_frac": 0.0,
            }
        )
        return out

    if gt.sum() == 0:
        return out

    dist_to_gt = ndimage.distance_transform_edt(
        np.logical_not(gt), sampling=spacing
    ).astype(np.float32)

    fp_dist = dist_to_gt[fp]
    out["fp_dist_median_mm"] = float(np.median(fp_dist))
    out["fp_dist_p95_mm"] = _nan_percentile(fp_dist, 95.0)

    out["fp_near_voxels_frac"] = float(np.mean(fp_dist <= float(near_mm)))
    out["fp_far_voxels_frac"] = float(np.mean(fp_dist > float(far_mm)))

    fp_lab, fp_n = _label_cc(fp)
    vox_vol = _voxel_volume_mm3(spacing)

    boundary_cc = 0
    blob_cc = 0
    boundary_vol = 0.0
    blob_vol = 0.0

    for cc_id in range(1, fp_n + 1):
        cc_mask = fp_lab == cc_id
        if not np.any(cc_mask):
            continue

        cc_min_dist = float(np.min(dist_to_gt[cc_mask]))
        cc_vol = float(int(cc_mask.sum())) * vox_vol

        if cc_min_dist <= float(near_mm):
            boundary_cc += 1
            boundary_vol += cc_vol
        else:
            blob_cc += 1
            blob_vol += cc_vol

    total_cc = float(boundary_cc + blob_cc)
    total_vol = float(boundary_vol + blob_vol)

    out["fp_boundary_cc"] = float(boundary_cc)
    out["fp_blob_cc"] = float(blob_cc)
    out["fp_boundary_volume_mm3"] = float(boundary_vol)
    out["fp_blob_volume_mm3"] = float(blob_vol)

    if total_cc > 0.0:
        out["fp_blob_cc_frac"] = float(blob_cc) / (total_cc + EPS)
    if total_vol > 0.0:
        out["fp_blob_volume_frac"] = float(blob_vol) / (total_vol + EPS)

    return out


def mean_entropy_in_mask(prob: np.ndarray, mask: np.ndarray) -> float:
    """Mean Bernoulli entropy inside a mask."""
    if mask.sum() == 0:
        return float("nan")

    p = prob[mask].astype(np.float32)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return float("nan")

    p = np.clip(p, EPS, 1.0 - EPS)
    ent = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))
    return float(np.mean(ent))


def brier_score(prob: np.ndarray, gt: np.ndarray, roi: np.ndarray) -> float:
    if roi.sum() == 0:
        return float("nan")
    p = prob[roi].astype(np.float32)
    y = gt[roi].astype(np.float32)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    prob: np.ndarray,
    gt: np.ndarray,
    roi: np.ndarray,
    n_bins: int,
    max_points: int,
    seed: int,
) -> float:
    """ECE for binary classification within ROI."""
    idx = np.flatnonzero(roi.ravel())
    if idx.size == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    if idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)

    p = prob.ravel()[idx].astype(np.float32)
    y = gt.ravel()[idx].astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = float(bins[i]), float(bins[i + 1])
        in_bin = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo)
        if not np.any(in_bin):
            continue
        conf = float(np.mean(p[in_bin]))
        acc = float(np.mean(y[in_bin]))
        w = float(np.mean(in_bin))
        ece += w * abs(acc - conf)

    return float(ece)


def compute_case_metrics(
    gt: np.ndarray,
    pred: np.ndarray,
    spacing: Tuple[float, float, float],
    prob: Optional[np.ndarray],
    small_voxels_thresh: int,
    ece_bins: int,
    ece_max_points: int,
    ece_seed: int,
    fp_near_mm: float,
    fp_far_mm: float,
    metrics_mode: str,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    metrics["dice"] = dice_score(gt, pred)
    metrics["hd95_mm"] = hd95_mm(gt, pred, spacing)

    lesion_metrics = lesion_detection_metrics(
        gt=gt,
        pred=pred,
        spacing=spacing,
        small_voxels_thresh=small_voxels_thresh,
    )
    metrics["small_lesion_recall"] = lesion_metrics[
        "small_lesion_recall"
    ]
    metrics["lesion_f1"] = lesion_metrics["lesion_f1"]
    metrics["fn_lesion_count"] = lesion_metrics["fn_lesion_count"]
    metrics["fn_volume_fraction"] = lesion_metrics[
        "fn_volume_fraction"
    ]

    fp_metrics = fp_error_profile(
        gt=gt,
        pred=pred,
        spacing=spacing,
        near_mm=float(fp_near_mm),
        far_mm=float(fp_far_mm),
    )
    metrics["fp_volume_mm3"] = fp_metrics["fp_volume_mm3"]

    if metrics_mode == "main":
        return metrics

    metrics["assd_mm"] = assd_mm(gt, pred, spacing)
    metrics.update(lesion_metrics)
    metrics.update(fp_metrics)
    metrics.update(
        fn_error_profile(
            gt=gt,
            pred=pred,
            spacing=spacing,
        )
    )

    if prob is None:
        metrics["mean_entropy_gt"] = float("nan")
        metrics["brier_union"] = float("nan")
        metrics["ece_union"] = float("nan")
        return metrics

    roi = np.logical_or(gt, pred)
    metrics["mean_entropy_gt"] = mean_entropy_in_mask(prob, gt)
    metrics["brier_union"] = brier_score(prob, gt, roi)
    metrics["ece_union"] = expected_calibration_error(
        prob=prob,
        gt=gt,
        roi=roi,
        n_bins=ece_bins,
        max_points=ece_max_points,
        seed=ece_seed,
    )
    return metrics


def extract_model_and_fold(
    per_case_csv: str,
) -> Tuple[str, Union[int, str]]:
    """Extract model name and fold from a per-case CSV path."""
    match = re.search(
        r"multiscale_eval_(.+?)_fold(\d+|all)(?:_(main|all))?\.csv$",
        Path(per_case_csv).name,
    )
    if not match:
        raise ValueError(
            "Cannot extract model_name and fold from per_case_csv: "
            f"{per_case_csv}"
        )

    fold_str = match.group(2)
    fold = int(fold_str) if fold_str.isdigit() else fold_str

    return match.group(1), fold


def format_json_value(value: float) -> float | str | None:
    """Round finite floats to 5 decimals; keep NaN/Inf as None."""
    if isinstance(value, (np.floating, float)):
        if math.isfinite(float(value)):
            return round(float(value), 5)
        return None
    if isinstance(value, (np.integer, int)):
        return float(value)
    return value


def build_summary_json(
    per_case_csv: Path,
    summary: pd.Series,
    df: pd.DataFrame,
) -> Dict[str, float | int | str | None]:
    """Build flat JSON summary in the requested format."""
    per_case_csv_str = str(per_case_csv)
    model_name, fold = extract_model_and_fold(per_case_csv_str)

    out: Dict[str, float | int | str | None] = {
        "per_case_csv": per_case_csv_str,
        "model_name": model_name,
        "fold": fold,
    }

    for key, value in summary.to_dict().items():
        out[key] = format_json_value(value)

    metrics_mode = "main"
    if "fn_volume_mm3" in df.columns:
        metrics_mode = "all"

    if metrics_mode == "all" and "fn_lesion_count" in df.columns:
        out["total_fn_lesion_count"] = format_json_value(
            float(df["fn_lesion_count"].sum())
        )

    if metrics_mode == "all" and "fn_volume_mm3" in df.columns:
        out["total_fn_volume_mm3"] = format_json_value(
            float(df["fn_volume_mm3"].sum())
        )

    if metrics_mode == "all":
        for key in [
            "fn_tiny_count",
            "fn_small_count",
            "fn_medium_count",
            "fn_large_count",
            "fn_tiny_volume_mm3",
            "fn_small_volume_mm3",
            "fn_medium_volume_mm3",
            "fn_large_volume_mm3",
        ]:
            if key in df.columns:
                out[f"total_{key}"] = format_json_value(
                    float(df[key].sum())
                )

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute multiscale segmentation metrics and save CSV."
    )
    p.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Folder with ground-truth masks (*.nii.gz).",
    )
    p.add_argument(
        "--pred_mask_dir",
        type=str,
        required=True,
        help="Folder with predicted masks (*.nii.gz).",
    )
    p.add_argument(
        "--pred_prob_dir",
        type=str,
        default=None,
        help="Folder with predicted probabilities (*.npz). Optional.",
    )
    p.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV path for per-case metrics.",
    )
    p.add_argument(
        "--out_json",
        type=str,
        default=None,
        help="Optional output JSON path for mean metrics summary.",
    )
    p.add_argument(
        "--allow_missing_prob",
        action="store_true",
        help="Allow missing npz probabilities (fills NaN columns).",
    )
    p.add_argument(
        "--small_voxels_thresh",
        type=int,
        default=50,
        help="Threshold (in voxels) to define small GT lesions.",
    )
    p.add_argument(
        "--fp_near_mm",
        type=float,
        default=2.0,
        help=(
            "FP components with min distance to GT <= this threshold "
            "(mm) are counted as boundary extensions."
        ),
    )
    p.add_argument(
        "--fp_far_mm",
        type=float,
        default=5.0,
        help=(
            "FP voxels with distance to GT > this threshold (mm) are "
            "counted as far (detached) FP."
        ),
    )
    p.add_argument(
        "--ece_bins",
        type=int,
        default=15,
        help="Number of bins for ECE.",
    )
    p.add_argument(
        "--ece_max_points",
        type=int,
        default=200000,
        help="Max voxels sampled for ECE to control runtime.",
    )
    p.add_argument(
        "--ece_seed",
        type=int,
        default=123,
        help="Random seed used for ECE subsampling.",
    )
    p.add_argument(
        "--metrics_mode",
        type=str,
        default="main",
        choices=["main", "all"],
        help=(
            "Metric set to compute and save. 'main' saves only the main "
            "paper metrics, while 'all' saves the full evaluation profile."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Metrics mode enabled: {args.metrics_mode}")

    gt_dir = Path(args.gt_dir)
    pred_mask_dir = Path(args.pred_mask_dir)
    pred_prob_dir = Path(args.pred_prob_dir) if args.pred_prob_dir else None

    cases = build_case_table(
        gt_dir=gt_dir,
        pred_mask_dir=pred_mask_dir,
        pred_prob_dir=pred_prob_dir,
        allow_missing_prob=bool(args.allow_missing_prob),
    )

    rows: List[Dict[str, float | str]] = []
    for case in cases:
        gt, spacing = load_nii_mask(case.gt_path)
        pred, _ = load_nii_mask(case.pred_mask_path)

        if gt.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch for case '{case.case_id}': "
                f"gt={gt.shape} pred={pred.shape}"
            )

        prob: Optional[np.ndarray] = None
        if case.pred_prob_path is not None:
            prob = load_npz_probability(case.pred_prob_path)
            if prob.shape != gt.shape:
                raise ValueError(
                    f"Probability shape mismatch for case '{case.case_id}': "
                    f"prob={prob.shape} gt={gt.shape}"
                )

        metrics = compute_case_metrics(
            gt=gt,
            pred=pred,
            spacing=spacing,
            prob=prob,
            small_voxels_thresh=int(args.small_voxels_thresh),
            ece_bins=int(args.ece_bins),
            ece_max_points=int(args.ece_max_points),
            ece_seed=int(args.ece_seed),
            fp_near_mm=float(args.fp_near_mm),
            fp_far_mm=float(args.fp_far_mm),
            metrics_mode=str(args.metrics_mode),
        )

        row: Dict[str, float | str] = {
            "case_id": case.case_id,
        }
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv_path = Path(args.out_csv)
    out_csv_suffix = out_csv_path.suffix
    out_csv_stem = out_csv_path.stem
    out_csv_path = out_csv_path.with_name(
        f"{out_csv_stem}_{args.metrics_mode}{out_csv_suffix}"
    )
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False, quoting=csv.QUOTE_MINIMAL)

    numeric_cols = [c for c in df.columns if c != "case_id"]
    summary = df[numeric_cols].mean(numeric_only=True)

    print(f"Saved per-case metrics to: {out_csv_path}")


    out_json_path = Path(args.out_json)
    out_json_suffix = out_json_path.suffix
    out_json_stem = out_json_path.stem
    out_json_path = out_json_path.with_name(
        f"{out_json_stem}_{args.metrics_mode}{out_json_suffix}"
    )
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    summary_json = build_summary_json(
        per_case_csv=out_csv_path,
        summary=summary,
        df=df,
    )

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print(f"Saved mean metrics JSON to: {out_json_path}")


if __name__ == "__main__":
    main()