#!/usr/bin/env python3
"""
Lesion-level feature analysis for paired nnUNet-style feature maps.

This script compares two models, for example:
- CATMIL
- nnUNet

For each case, it:
1. loads feature maps from both models, shape (C, X, Y, Z),
2. loads a ground-truth mask,
3. resamples the GT to the feature-map space if needed,
4. computes connected components on the aligned GT,
5. extracts one lesion embedding per lesion per model,
6. optionally loads prediction masks and labels each lesion as TP/FN,
7. computes background centroids and lesion-to-background distances,
8. saves lesion-level tables,
9. runs UMAP and t-SNE on lesion embeddings,
10. saves plots and numeric outputs.

Expected feature layout:
    <feature_root>/<case_id>/<layer_dir>/feature_map.npy

Example:
    features/CATMIL/P13_T1/decoder_seg_layers_4/feature_map.npy
    features/nnUNet/P13_T1/decoder_seg_layers_4/feature_map.npy

GT layout:
    <gt_root>/<case_id>.nii.gz

Prediction layout (optional):
    <pred_root>/<case_id>.nii.gz

Notes
-----
- Best practice is to use GT already in nnUNet preprocessed space.
- If GT is not in feature-map space, this script resamples it with nearest
  neighbor to match the feature-map shape.
- UMAP and t-SNE are qualitative. The saved CSV contains quantitative metrics
  as well.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label, zoom
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import umap  # type: ignore
except ImportError as exc:
    raise ImportError(
        "umap-learn is required. Install with: pip install umap-learn"
    ) from exc


@dataclass
class Config:
    catmil_feature_root: Path
    nnunet_feature_root: Path
    gt_root: Path
    output_root: Path
    layer_dir: str
    catmil_pred_root: Optional[Path]
    nnunet_pred_root: Optional[Path]
    gt_suffix: str
    pred_suffix: str
    min_lesion_voxels: int
    small_lesion_threshold: int
    lesion_overlap_threshold: float
    n_bg_samples_per_case: int
    random_seed: int
    run_tsne: bool
    run_umap: bool
    umap_n_neighbors: int
    umap_min_dist: float
    tsne_perplexity: float


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Lesion-level feature analysis for CATMIL vs nnUNet."
    )
    parser.add_argument(
        "--catmil_feature_root",
        type=Path,
        required=True,
        help="Root folder of CATMIL extracted features.",
    )
    parser.add_argument(
        "--nnunet_feature_root",
        type=Path,
        required=True,
        help="Root folder of nnUNet extracted features.",
    )
    parser.add_argument(
        "--gt_root",
        type=Path,
        required=True,
        help="Folder containing GT masks as NIfTI files.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        required=True,
        help="Folder where outputs will be saved.",
    )
    parser.add_argument(
        "--layer_dir",
        type=str,
        required=True,
        help="Layer directory under each case, for example "
        "'decoder_seg_layers_4'.",
    )
    parser.add_argument(
        "--catmil_pred_root",
        type=Path,
        default=None,
        help="Optional CATMIL prediction folder.",
    )
    parser.add_argument(
        "--nnunet_pred_root",
        type=Path,
        default=None,
        help="Optional nnUNet prediction folder.",
    )
    parser.add_argument(
        "--gt_suffix",
        type=str,
        default=".nii.gz",
        help="GT file suffix. Default: .nii.gz",
    )
    parser.add_argument(
        "--pred_suffix",
        type=str,
        default=".nii.gz",
        help="Prediction file suffix. Default: .nii.gz",
    )
    parser.add_argument(
        "--min_lesion_voxels",
        type=int,
        default=3,
        help="Minimum lesion size to keep after connected components.",
    )
    parser.add_argument(
        "--small_lesion_threshold",
        type=int,
        default=50,
        help="Lesions below this size are labeled as small.",
    )
    parser.add_argument(
        "--lesion_overlap_threshold",
        type=float,
        default=0.0,
        help="A GT lesion is TP if overlap ratio with prediction is greater "
        "than this threshold.",
    )
    parser.add_argument(
        "--n_bg_samples_per_case",
        type=int,
        default=5000,
        help="Number of background voxels sampled per case.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--disable_tsne",
        action="store_true",
        help="Disable t-SNE.",
    )
    parser.add_argument(
        "--disable_umap",
        action="store_true",
        help="Disable UMAP.",
    )
    parser.add_argument(
        "--umap_n_neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors.",
    )
    parser.add_argument(
        "--umap_min_dist",
        type=float,
        default=0.1,
        help="UMAP min_dist.",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity.",
    )
    args = parser.parse_args()

    return Config(
        catmil_feature_root=args.catmil_feature_root,
        nnunet_feature_root=args.nnunet_feature_root,
        gt_root=args.gt_root,
        output_root=args.output_root,
        layer_dir=args.layer_dir,
        catmil_pred_root=args.catmil_pred_root,
        nnunet_pred_root=args.nnunet_pred_root,
        gt_suffix=args.gt_suffix,
        pred_suffix=args.pred_suffix,
        min_lesion_voxels=args.min_lesion_voxels,
        small_lesion_threshold=args.small_lesion_threshold,
        lesion_overlap_threshold=args.lesion_overlap_threshold,
        n_bg_samples_per_case=args.n_bg_samples_per_case,
        random_seed=args.random_seed,
        run_tsne=not args.disable_tsne,
        run_umap=not args.disable_umap,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        tsne_perplexity=args.tsne_perplexity,
    )


def maybe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def display_model_name(model_key: str) -> str:
    """Return user-facing model names for plots and saved outputs."""
    mapping = {
        "CATMIL": "nnUNet-CATMIL",
        "nnUNet": "nnUNet-DiceCE",
    }
    return mapping.get(model_key, model_key)


def find_case_ids(
    catmil_root: Path,
    nnunet_root: Path,
    layer_dir: str,
) -> List[str]:
    cat_cases = {
        p.name
        for p in catmil_root.iterdir()
        if p.is_dir() and (p / layer_dir / "feature_map.npy").is_file()
    }
    nn_cases = {
        p.name
        for p in nnunet_root.iterdir()
        if p.is_dir() and (p / layer_dir / "feature_map.npy").is_file()
    }
    return sorted(cat_cases & nn_cases)


def load_feature_map(feature_root: Path, case_id: str, layer_dir: str) -> np.ndarray:
    path = feature_root / case_id / layer_dir / "feature_map.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature map: {path}")
    feat = np.load(path)
    if feat.ndim != 4:
        raise ValueError(
            f"Feature map for {case_id} must have shape (C, X, Y, Z), "
            f"got {feat.shape}"
        )
    return feat.astype(np.float32)


def load_nifti_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing NIfTI file: {path}")
    arr = nib.load(str(path)).get_fdata()
    return np.asarray(arr)


def load_mask(mask_root: Path, case_id: str, suffix: str) -> np.ndarray:
    path = mask_root / f"{case_id}{suffix}"
    arr = load_nifti_array(path)
    return (arr > 0).astype(np.uint8)


def align_binary_mask_to_shape(
    mask: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> np.ndarray:
    if mask.shape == target_shape:
        return mask.astype(np.uint8)

    zoom_factors = tuple(
        float(t) / float(s) for s, t in zip(mask.shape, target_shape)
    )
    aligned = zoom(mask.astype(np.float32), zoom=zoom_factors, order=0)
    aligned = (aligned > 0.5).astype(np.uint8)

    if aligned.shape != target_shape:
        raise ValueError(
            f"Aligned mask shape mismatch. Expected {target_shape}, "
            f"got {aligned.shape}"
        )
    return aligned


def component_labeling(
    gt_mask: np.ndarray,
    min_lesion_voxels: int,
) -> Tuple[np.ndarray, List[int]]:
    labeled, num = label(gt_mask)
    keep_ids: List[int] = []

    for lesion_id in range(1, num + 1):
        lesion_voxels = int(np.sum(labeled == lesion_id))
        if lesion_voxels >= min_lesion_voxels:
            keep_ids.append(lesion_id)

    filtered = np.zeros_like(labeled, dtype=np.int32)
    for new_id, old_id in enumerate(keep_ids, start=1):
        filtered[labeled == old_id] = new_id

    return filtered, list(range(1, len(keep_ids) + 1))


def sample_background_embeddings(
    feature_map: np.ndarray,
    gt_mask: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    bg_mask = gt_mask == 0
    coords = np.argwhere(bg_mask)

    if coords.size == 0:
        raise ValueError("No background voxels found.")

    n_take = min(n_samples, coords.shape[0])
    idx = rng.choice(coords.shape[0], size=n_take, replace=False)
    coords = coords[idx]

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    emb = feature_map[:, x, y, z].T
    return emb.astype(np.float32), coords.astype(np.int32)


def lesion_overlap_ratio(
    lesion_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> float:
    lesion_voxels = int(np.sum(lesion_mask))
    if lesion_voxels == 0:
        return 0.0
    overlap = int(np.sum(lesion_mask & pred_mask))
    return float(overlap) / float(lesion_voxels)


def detection_label(
    lesion_mask: np.ndarray,
    pred_mask: Optional[np.ndarray],
    overlap_threshold: float,
) -> str:
    if pred_mask is None:
        return "unknown"

    ratio = lesion_overlap_ratio(lesion_mask, pred_mask)
    return "TP" if ratio > overlap_threshold else "FN"


def compute_embedding_from_mask(
    feature_map: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    vox = feature_map[:, mask]
    if vox.shape[1] == 0:
        raise ValueError("Empty lesion mask when computing embedding.")
    return vox.mean(axis=1).astype(np.float32)


def compute_centroid_from_mask(mask: np.ndarray) -> Tuple[float, float, float]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return (math.nan, math.nan, math.nan)
    ctr = coords.mean(axis=0)
    return (float(ctr[0]), float(ctr[1]), float(ctr[2]))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return math.nan
    sim = float(np.dot(a, b) / (na * nb))
    return float(1.0 - sim)


def plot_embedding(
    emb_2d: np.ndarray,
    labels: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    labels = np.asarray(labels)
    uniq = sorted(np.unique(labels))

    plt.figure(figsize=(8, 6))
    for lab in uniq:
        idx = labels == lab
        plt.scatter(
            emb_2d[idx, 0],
            emb_2d[idx, 1],
            s=18,
            alpha=0.75,
            label=str(lab),
        )
    plt.title(title)
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.legend(markerscale=1.5, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_umap_projection(
    x: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    random_seed: int,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=random_seed,
    )
    return reducer.fit_transform(x)


def run_tsne_projection(
    x: np.ndarray,
    perplexity: float,
    random_seed: int,
) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_seed,
    )
    return tsne.fit_transform(x)


def build_analysis_tables(
    config: Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config.random_seed)

    lesion_rows: List[Dict] = []
    bg_rows: List[Dict] = []

    case_ids = find_case_ids(
        config.catmil_feature_root,
        config.nnunet_feature_root,
        config.layer_dir,
    )

    if len(case_ids) == 0:
        raise RuntimeError("No overlapping cases found.")

    for case_id in case_ids:
        print(f"Processing case: {case_id}")

        feat_cat = load_feature_map(
            config.catmil_feature_root, case_id, config.layer_dir
        )
        feat_nnu = load_feature_map(
            config.nnunet_feature_root, case_id, config.layer_dir
        )

        if feat_cat.shape != feat_nnu.shape:
            raise ValueError(
                f"Feature shape mismatch for {case_id}: "
                f"{feat_cat.shape} vs {feat_nnu.shape}"
            )

        spatial_shape = feat_cat.shape[1:]
        gt_raw = load_mask(config.gt_root, case_id, config.gt_suffix)
        gt = align_binary_mask_to_shape(gt_raw, spatial_shape)

        pred_cat = None
        if config.catmil_pred_root is not None:
            pred_cat_raw = load_mask(
                config.catmil_pred_root, case_id, config.pred_suffix
            )
            pred_cat = align_binary_mask_to_shape(pred_cat_raw, spatial_shape)

        pred_nnu = None
        if config.nnunet_pred_root is not None:
            pred_nnu_raw = load_mask(
                config.nnunet_pred_root, case_id, config.pred_suffix
            )
            pred_nnu = align_binary_mask_to_shape(pred_nnu_raw, spatial_shape)

        labeled_gt, lesion_ids = component_labeling(
            gt_mask=gt,
            min_lesion_voxels=config.min_lesion_voxels,
        )

        bg_cat, _ = sample_background_embeddings(
            feat_cat,
            gt,
            config.n_bg_samples_per_case,
            rng,
        )
        bg_nnu, _ = sample_background_embeddings(
            feat_nnu,
            gt,
            config.n_bg_samples_per_case,
            rng,
        )

        bg_centroid_cat = bg_cat.mean(axis=0)
        bg_centroid_nnu = bg_nnu.mean(axis=0)

        bg_rows.append(
            {
                "case_id": case_id,
                "model": display_model_name("CATMIL"),
                "n_bg_samples": int(bg_cat.shape[0]),
                **{
                    f"f{i:02d}": float(bg_centroid_cat[i])
                    for i in range(bg_centroid_cat.shape[0])
                },
            }
        )
        bg_rows.append(
            {
                "case_id": case_id,
                "model": display_model_name("nnUNet"),
                "n_bg_samples": int(bg_nnu.shape[0]),
                **{
                    f"f{i:02d}": float(bg_centroid_nnu[i])
                    for i in range(bg_centroid_nnu.shape[0])
                },
            }
        )

        for lesion_id in lesion_ids:
            lesion_mask = labeled_gt == lesion_id
            lesion_voxels = int(np.sum(lesion_mask))
            size_group = (
                "small"
                if lesion_voxels < config.small_lesion_threshold
                else "large"
            )
            ctr_x, ctr_y, ctr_z = compute_centroid_from_mask(lesion_mask)

            emb_cat = compute_embedding_from_mask(feat_cat, lesion_mask)
            emb_nnu = compute_embedding_from_mask(feat_nnu, lesion_mask)

            det_cat = detection_label(
                lesion_mask=lesion_mask,
                pred_mask=pred_cat,
                overlap_threshold=config.lesion_overlap_threshold,
            )
            det_nnu = detection_label(
                lesion_mask=lesion_mask,
                pred_mask=pred_nnu,
                overlap_threshold=config.lesion_overlap_threshold,
            )

            row_common = {
                "case_id": case_id,
                "lesion_id": lesion_id,
                "lesion_voxels": lesion_voxels,
                "size_group": size_group,
                "centroid_x": ctr_x,
                "centroid_y": ctr_y,
                "centroid_z": ctr_z,
            }

            lesion_rows.append(
                {
                    **row_common,
                    "model": display_model_name("CATMIL"),
                    "detection": det_cat,
                    "dist_to_bg_euclidean": euclidean_distance(
                        emb_cat, bg_centroid_cat
                    ),
                    "dist_to_bg_cosine": cosine_distance(
                        emb_cat, bg_centroid_cat
                    ),
                    **{
                        f"f{i:02d}": float(emb_cat[i])
                        for i in range(emb_cat.shape[0])
                    },
                }
            )

            lesion_rows.append(
                {
                    **row_common,
                    "model": display_model_name("nnUNet"),
                    "detection": det_nnu,
                    "dist_to_bg_euclidean": euclidean_distance(
                        emb_nnu, bg_centroid_nnu
                    ),
                    "dist_to_bg_cosine": cosine_distance(
                        emb_nnu, bg_centroid_nnu
                    ),
                    **{
                        f"f{i:02d}": float(emb_nnu[i])
                        for i in range(emb_nnu.shape[0])
                    },
                }
            )

    lesion_df = pd.DataFrame(lesion_rows)
    bg_df = pd.DataFrame(bg_rows)
    return lesion_df, bg_df


def save_summary_tables(
    lesion_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    summary = (
        lesion_df.groupby(["model", "detection", "size_group"])
        .agg(
            n_lesions=("lesion_id", "count"),
            mean_dist_bg_euclidean=("dist_to_bg_euclidean", "mean"),
            mean_dist_bg_cosine=("dist_to_bg_cosine", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "lesion_summary.csv", index=False)

    paired = (
        lesion_df.pivot_table(
            index=["case_id", "lesion_id", "lesion_voxels", "size_group"],
            columns="model",
            values=["dist_to_bg_euclidean", "dist_to_bg_cosine", "detection"],
            aggfunc="first",
        )
        .reset_index()
    )
    paired.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in paired.columns
    ]
    paired.to_csv(out_dir / "paired_lesion_summary.csv", index=False)


def make_embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    feat_cols = sorted([c for c in df.columns if c.startswith("f")])
    x = df[feat_cols].to_numpy(dtype=np.float32)
    return x


def run_and_save_embeddings(
    lesion_df: pd.DataFrame,
    out_dir: Path,
    config: Config,
) -> None:
    x = make_embedding_matrix(lesion_df)
    x_scaled = StandardScaler().fit_transform(x)

    if config.run_umap:
        umap_xy = run_umap_projection(
            x_scaled,
            n_neighbors=config.umap_n_neighbors,
            min_dist=config.umap_min_dist,
            random_seed=config.random_seed,
        )
        umap_df = lesion_df.copy()
        umap_df["umap_x"] = umap_xy[:, 0]
        umap_df["umap_y"] = umap_xy[:, 1]
        umap_df.to_csv(out_dir / "lesion_embeddings_umap.csv", index=False)

        plot_embedding(
            emb_2d=umap_xy,
            labels=umap_df["model"].tolist(),
            out_path=out_dir / "umap_by_model.png",
            title="UMAP by model",
        )
        plot_embedding(
            emb_2d=umap_xy,
            labels=umap_df["detection"].tolist(),
            out_path=out_dir / "umap_by_detection.png",
            title="UMAP by detection",
        )
        plot_embedding(
            emb_2d=umap_xy,
            labels=umap_df["size_group"].tolist(),
            out_path=out_dir / "umap_by_size.png",
            title="UMAP by lesion size",
        )

        combined_label = [
            f"{m}_{d}"
            for m, d in zip(
                umap_df["model"].tolist(),
                umap_df["detection"].tolist(),
            )
        ]
        plot_embedding(
            emb_2d=umap_xy,
            labels=combined_label,
            out_path=out_dir / "umap_by_model_detection.png",
            title="UMAP by model and detection",
        )

    if config.run_tsne:
        tsne_xy = run_tsne_projection(
            x_scaled,
            perplexity=config.tsne_perplexity,
            random_seed=config.random_seed,
        )
        tsne_df = lesion_df.copy()
        tsne_df["tsne_x"] = tsne_xy[:, 0]
        tsne_df["tsne_y"] = tsne_xy[:, 1]
        tsne_df.to_csv(out_dir / "lesion_embeddings_tsne.csv", index=False)

        plot_embedding(
            emb_2d=tsne_xy,
            labels=tsne_df["model"].tolist(),
            out_path=out_dir / "tsne_by_model.png",
            title="t-SNE by model",
        )
        plot_embedding(
            emb_2d=tsne_xy,
            labels=tsne_df["detection"].tolist(),
            out_path=out_dir / "tsne_by_detection.png",
            title="t-SNE by detection",
        )
        plot_embedding(
            emb_2d=tsne_xy,
            labels=tsne_df["size_group"].tolist(),
            out_path=out_dir / "tsne_by_size.png",
            title="t-SNE by lesion size",
        )


def main() -> None:
    config = parse_args()
    maybe_mkdir(config.output_root)

    save_json(
        config.output_root / "analysis_config.json",
        {
            "catmil_feature_root": str(config.catmil_feature_root),
            "nnunet_feature_root": str(config.nnunet_feature_root),
            "gt_root": str(config.gt_root),
            "output_root": str(config.output_root),
            "layer_dir": config.layer_dir,
            "catmil_pred_root": (
                None if config.catmil_pred_root is None
                else str(config.catmil_pred_root)
            ),
            "nnunet_pred_root": (
                None if config.nnunet_pred_root is None
                else str(config.nnunet_pred_root)
            ),
            "gt_suffix": config.gt_suffix,
            "pred_suffix": config.pred_suffix,
            "min_lesion_voxels": config.min_lesion_voxels,
            "small_lesion_threshold": config.small_lesion_threshold,
            "lesion_overlap_threshold": config.lesion_overlap_threshold,
            "n_bg_samples_per_case": config.n_bg_samples_per_case,
            "random_seed": config.random_seed,
            "run_tsne": config.run_tsne,
            "run_umap": config.run_umap,
            "umap_n_neighbors": config.umap_n_neighbors,
            "umap_min_dist": config.umap_min_dist,
            "tsne_perplexity": config.tsne_perplexity,
        },
    )

    lesion_df, bg_df = build_analysis_tables(config)

    lesion_df.to_csv(
        config.output_root / "lesion_embeddings.csv",
        index=False,
    )
    bg_df.to_csv(
        config.output_root / "background_centroids.csv",
        index=False,
    )

    save_summary_tables(lesion_df, config.output_root)
    run_and_save_embeddings(lesion_df, config.output_root, config)

    print("Done.")
    print(f"Saved outputs to: {config.output_root}")


if __name__ == "__main__":
    main()