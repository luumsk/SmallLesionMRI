from typing import Dict, Tuple, Optional, Callable, Sequence
from dataclasses import dataclass
from typing import Iterable

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

try:
    import nibabel as nib
except Exception:  # pragma: no cover
    nib = None

from scipy import ndimage as ndi
from scipy.ndimage import (
    label,
    generate_binary_structure,
    gaussian_filter,
    binary_erosion,
    binary_dilation,
    sobel,
)

# -------------------------------------------
# Utility functions for lesion analysis
# -------------------------------------------
def connected_components(mask,
                         rank: int = 3,
                         connectivity: int = 1) -> tuple[np.ndarray, int]:
    """
    Computes connected components in a 3D binary mask.

    Parameters:
    mask (numpy.ndarray): A 3D binary numpy array where non-zero values indicate the presence of a lesion.

    Returns:
    tuple: A tuple containing:
        - labeled_mask (numpy.ndarray): A 3D array where each connected component is assigned a unique integer label.
        - num_components (int): The number of connected components found in the mask.
    """
    structure = generate_binary_structure(rank=rank, connectivity=connectivity)  # 26-connectivity
    labeled_mask, num_components = label(mask, structure=structure)
    return labeled_mask, num_components


def show_slice_with_mask(image, mask, slice_index, axis=2):
    """Display a 2D slice of a 3D MRI volume and its mask.

    Assumes arrays are in (X, Y, Z) order:
    - axis=0: sagittal slice at x = slice_index
    - axis=1: coronal slice at y = slice_index
    - axis=2: axial slice at z = slice_index

    The display uses `.T` with `origin="lower"` for a consistent visual
    convention.
    """

    # Arrays are (X, Y, Z)
    if axis == 0:
        img_slice = image[slice_index, :, :]
        mask_slice = mask[slice_index, :, :]
    elif axis == 1:
        img_slice = image[:, slice_index, :]
        mask_slice = mask[:, slice_index, :]
    else:
        img_slice = image[:, :, slice_index]
        mask_slice = mask[:, :, slice_index]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_slice.T, cmap="gray", origin="lower")
    axes[0].axis("off")
    axes[0].set_title(f"MRI Slice {slice_index} (axis={axis})")

    axes[1].imshow(mask_slice.T, cmap="gray", origin="lower")
    axes[1].axis("off")
    axes[1].set_title(f"Segmentation Mask {slice_index} (axis={axis})")

    return fig, axes


def compute_lesion_volume(mask: np.ndarray,
                          zoom: tuple[float, float, float],
                          unit: str = "mL") -> float:
    """Compute total lesion volume from a 3D mask.

    Parameters
    ----------
    mask:
        3D array. Voxels with mask > 0 are counted as lesion.
    zoom:
        (sx, sy, sz) voxel spacing in millimeters.
    unit:
        "mm3" for cubic millimeters or "mL" for milliliters.

    Returns
    -------
    float
        Total lesion volume in the requested unit.
    """
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {mask.shape}")
    if len(zoom) != 3:
        raise ValueError("zoom must be a length-3 tuple (sx, sy, sz)")
    if any(z <= 0 for z in zoom):
        raise ValueError(f"zoom must be positive, got {zoom}")

    voxel_vol_mm3 = float(zoom[0]) * float(zoom[1]) * float(zoom[2])
    vol_mm3 = float(np.sum(mask > 0)) * voxel_vol_mm3

    if unit == "mm3":
        return vol_mm3
    if unit == "mL":
        return vol_mm3 / 1000.0

    raise ValueError("unit must be 'mm3' or 'mL'")


def compute_lesion_volumes_for_components(mask: np.ndarray,
                                       zoom: tuple[float, float, float],
                                       unit: str = "mL") -> dict[int, float]:
    """Compute lesion volumes for each connected component in a 3D mask.

    Parameters
    ----------
    mask:
        3D array. Voxels with mask > 0 are counted as lesion.
    zoom:
        (sx, sy, sz) voxel spacing in millimeters.
    unit:
        "mm3" for cubic millimeters or "mL" for milliliters.

    Returns
    -------
    dict[int, float]
        Dictionary mapping component labels to their respective volumes.
    """
    labeled_mask, num_components = connected_components(mask)
    volumes = {}
    for component_id in range(1, num_components + 1):
        component_mask = (labeled_mask == component_id).astype(np.uint8)
        volume = compute_lesion_volume(component_mask, zoom, unit)
        volumes[component_id] = volume
    return volumes


def zscore_normalize(
    image: np.ndarray,
    brain_mask: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score normalize MRI intensities using a brain mask.

    I_norm = (I - mean_brain) / std_brain
    """
    brain_vals = image[brain_mask > 0]

    mean = float(np.mean(brain_vals))
    std = float(np.std(brain_vals))

    return (image - mean) / (std + eps)

# -------------------------------------------
# Lesion contrast metrics
# -------------------------------------------
def mad(x: np.ndarray) -> float:
    """Median absolute deviation."""
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def lesion_background_ring(
    lesion_mask: np.ndarray,
    ring_mm: float,
    spacing_mm: tuple[float, float, float],
    brain_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Create a perilesional background ring mask.

    Notes
    -----
    The previous dilation-based implementation used `iterations=max(rad)` which
    can over/under-grow the ring when voxel spacing is anisotropic and can make
    the ring size depend on lesion size in voxel space.

    This implementation uses an exact Euclidean distance transform in
    millimeters (via `sampling=spacing_mm`) to create a ring of thickness
    `ring_mm` around the lesion.

    The ring is defined as voxels that are:
    - outside the lesion, and
    - within `ring_mm` of the lesion boundary, and
    - optionally inside `brain_mask`.
    """
    if lesion_mask.ndim != 3:
        raise ValueError(
            f"lesion_mask must be 3D, got shape {lesion_mask.shape}"
        )

    lesion_b = lesion_mask.astype(bool, copy=False)
    if int(lesion_b.sum()) == 0:
        return np.zeros_like(lesion_b, dtype=bool)

    if len(spacing_mm) != 3:
        raise ValueError(
            "spacing_mm must be a length-3 tuple (sx, sy, sz)"
        )
    if any(float(sp) <= 0 for sp in spacing_mm):
        raise ValueError(f"spacing_mm must be positive, got {spacing_mm}")

    if float(ring_mm) <= 0:
        raise ValueError(f"ring_mm must be > 0, got {ring_mm}")

    # Distance (in mm) to the nearest lesion voxel.
    dist_mm = ndi.distance_transform_edt(~lesion_b, sampling=spacing_mm)

    ring = (dist_mm > 0.0) & (dist_mm <= float(ring_mm))

    if brain_mask is not None:
        ring = ring & brain_mask.astype(bool, copy=False)

    return ring


def local_robust_cnr(
    image_norm: np.ndarray,
    lesion_mask: np.ndarray,
    ring_mask: np.ndarray,
    eps: float = 1e-8,
) -> float:
    """
    Robust lesion–background contrast (single scalar).
    """
    I_L = image_norm[lesion_mask > 0]
    I_B = image_norm[ring_mask > 0]

    # For tiny lesions/rings, medians and MAD can be unstable.
    # Return NaN to avoid interpreting noise as low/high contrast.
    if I_L.size < 10 or I_B.size < 50:
        return float("nan")

    if I_L.size == 0 or I_B.size == 0:
        return float("nan")

    return float(
        abs(np.median(I_L) - np.median(I_B)) / (mad(I_B) + eps)
    )


@dataclass(frozen=True)
class ContrastSelection:
    """Defines how lesions are selected for visualization.

    Notes
    -----
    The lesion IDs selected by thresholds (or provided via `lesion_ids`) must
    match the integer labels present in `lesion_label_map` (0 = background,
    1..N). If your `contrast` dict was computed from a different connected-
    component labeling, the visualized lesion will be wrong.
    """
    mode: str  # "low", "high", "range", "ids"
    low_thr: float = 1.0
    high_thr: float = 2.0
    min_val: float = -np.inf
    max_val: float = np.inf
    lesion_ids: tuple[int, ...] = ()


def _get_bbox(mask: np.ndarray, margin: int = 3) -> tuple[slice, slice, slice] | None:
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        return None

    # Arrays are (X, Y, Z)
    xmin = max(0, int(coords[0].min()) - margin)
    ymin = max(0, int(coords[1].min()) - margin)
    zmin = max(0, int(coords[2].min()) - margin)

    xmax = min(mask.shape[0], int(coords[0].max()) + margin + 1)
    ymax = min(mask.shape[1], int(coords[1].max()) + margin + 1)
    zmax = min(mask.shape[2], int(coords[2].max()) + margin + 1)

    return slice(xmin, xmax), slice(ymin, ymax), slice(zmin, zmax)


def _plot_orthogonal_views(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    title: str,
    ring_mask: np.ndarray | None = None,
) -> None:
    coords = np.where(lesion_mask > 0)
    if coords[0].size == 0:
        return

    # Arrays are (X, Y, Z) order.
    center = np.array(coords).mean(axis=1).astype(int)
    x, y, z = int(center[0]), int(center[1]), int(center[2])

    # axis=2 is axial (z)
    axial_img = image[:, :, z]
    axial_msk = lesion_mask[:, :, z]
    coronal_img = image[:, y, :]
    coronal_msk = lesion_mask[:, y, :]
    sagittal_img = image[x, :, :]
    sagittal_msk = lesion_mask[x, :, :]

    if ring_mask is not None:
        axial_ring = ring_mask[:, :, z]
        coronal_ring = ring_mask[:, y, :]
        sagittal_ring = ring_mask[x, :, :]
    else:
        axial_ring = coronal_ring = sagittal_ring = None

    views = [
        (sagittal_img, sagittal_msk, sagittal_ring, "Sagittal (x, axis=0)"),
        (coronal_img, coronal_msk, coronal_ring, "Coronal (y, axis=1)"),
        (axial_img, axial_msk, axial_ring, "Axial (z, axis=2)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(title)

    for ax, (img, msk, rsk, name) in zip(axes, views):
        ax.imshow(img.T, cmap="gray", origin="lower")
        if np.any(msk):
            ax.contour(msk.T, colors="red", linewidths=1)
        if rsk is not None and np.any(rsk):
            ax.contour(rsk.T, colors="yellow", linewidths=1)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def _select_lesions(
    contrast: dict[int, float],
    selection: ContrastSelection,
) -> list[int]:
    if selection.mode == "ids":
        return [lid for lid in selection.lesion_ids if lid in contrast]

    if selection.mode == "low":
        return [lid for lid, v in contrast.items() if v < selection.low_thr]

    if selection.mode == "high":
        return [lid for lid, v in contrast.items() if v >= selection.high_thr]

    if selection.mode == "range":
        return [
            lid
            for lid, v in contrast.items()
            if (selection.min_val <= v < selection.max_val)
        ]

    raise ValueError(
        "selection.mode must be one of: 'low', 'high', 'range', 'ids'."
    )


def _available_label_ids(lesion_label_map: np.ndarray) -> set[int]:
    """Return the set of positive label IDs present in the label map."""
    labels = np.unique(lesion_label_map)
    labels = labels[labels > 0]
    return set(int(x) for x in labels)


#
# Debug tip: If a lesion looks wrong, you likely have an ID mismatch between
# `contrast` and `lesion_label_map`. Use `show_full_context=True` to verify the
# lesion on the full image. Pass `ring_mask` to overlay the LCNR background ring.
# Convention: arrays are (X, Y, Z) and axial is axis=2.
def visualize_lesion_contrast(
    image_norm: np.ndarray,
    lesion_label_map: np.ndarray,
    contrast: dict[int, float],
    selection: ContrastSelection,
    max_lesions: int | None = None,
    sort_by_contrast: bool = True,
    margin: int = 3,
    show_full_context: bool = False,
    ring_mask: np.ndarray | None = None,
) -> None:
    """
    Visualize lesions selected by contrast group or explicit IDs.

    Parameters
    ----------
    image_norm:
        Normalized 3D MRI image (recommended: brain-masked z-score).
    lesion_label_map:
        3D int label map from connected components (0 = background, 1..N).
    contrast:
        Dict mapping lesion_id -> scalar contrast value (e.g., LCNR).
    selection:
        ContrastSelection defining which lesions to visualize.
    max_lesions:
        Optional cap on number of lesions displayed.
    sort_by_contrast:
        If True, sort selected lesions by contrast (ascending).
    margin:
        Margin (in voxels) around lesion bounding box for cropping.
    show_full_context:
        If True, show a full-FOV axial slice at the lesion centroid to verify
        the ID/selection.
    ring_mask:
        Optional 3D ring mask (same shape) to overlay (yellow) for sanity
        checking LCNR background sampling.
    """
    lesion_ids = _select_lesions(contrast, selection)

    if image_norm.shape != lesion_label_map.shape:
        raise ValueError(
            "image_norm and lesion_label_map must have the same shape, "
            f"got {image_norm.shape} vs {lesion_label_map.shape}"
        )

    if ring_mask is not None and ring_mask.shape != image_norm.shape:
        raise ValueError(
            "ring_mask must have the same shape as image_norm, "
            f"got {ring_mask.shape} vs {image_norm.shape}"
        )

    present_ids = _available_label_ids(lesion_label_map)
    missing_in_map = [lid for lid in lesion_ids if lid not in present_ids]
    if missing_in_map:
        print(
            "Warning: selected lesion IDs not present in lesion_label_map: "
            f"{missing_in_map}. This usually means contrast IDs and label-map "
            "IDs are from different connected-component labelings."
        )

    lesion_ids = [lid for lid in lesion_ids if lid in present_ids]

    if not lesion_ids:
        print("No lesions matched the selection.")
        return

    if sort_by_contrast:
        lesion_ids = sorted(lesion_ids, key=lambda lid: contrast[lid])

    if max_lesions is not None:
        lesion_ids = lesion_ids[:max_lesions]

    for lid in lesion_ids:
        lesion_mask = lesion_label_map == lid

        if show_full_context:
            coords = np.where(lesion_mask > 0)
            if coords[0].size > 0:
                zc = int(np.round(coords[2].mean()))
                fig, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.imshow(image_norm[:, :, zc].T, cmap="gray", origin="lower")
                ax.contour(lesion_mask[:, :, zc].T, colors="red", linewidths=1)
                ax.set_title(f"Full FOV axial (z={zc}, axis=2) | Lesion {lid}")
                ax.axis("off")
                plt.tight_layout()
                plt.show()

        bbox = _get_bbox(lesion_mask, margin=margin)
        if bbox is None:
            continue

        img_crop = image_norm[bbox]
        mask_crop = lesion_mask[bbox]
        val = contrast.get(lid, float("nan"))

        ring_crop = None
        if ring_mask is not None:
            ring_crop = ring_mask[bbox]

        _plot_orthogonal_views(
            img_crop,
            mask_crop,
            title=(
                f"Lesion {lid} | Contrast = {val:.3f} | Voxels={int(mask_crop.sum())}"
            ),
            ring_mask=ring_crop,
        )


# -------------------------------------------
# Lesion boundary sharpness metrics
# -------------------------------------------
def boundary_band_gradient_sharpness(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma_mm: float = 1.0,
    band_iters: int = 1,
    summary: str = "median",
) -> Dict[str, float]:
    """Compute lesion-boundary sharpness using gradient magnitude in a band.

    This metric samples the image gradient magnitude in a thin band around the
    lesion boundary (inside + outside). Higher values generally indicate a
    sharper boundary (stronger intensity change across the edge).

    IMPORTANT
    ---------
    - For a *single-lesion* sharpness, `mask` must contain only that lesion.
      (e.g., `mask = (lesion_label_map == lesion_id)`.)
    - To compare across subjects/scans, prefer using an intensity-normalized
      image (e.g., brain-masked z-score) as `image`.

    Parameters
    ----------
    image:
        3D image array in (X, Y, Z) order.
    mask:
        3D binary mask of ONE lesion.
    spacing:
        (sx, sy, sz) voxel spacing in millimeters, matching (X, Y, Z).
    sigma_mm:
        Optional Gaussian smoothing in millimeters to reduce noise before
        gradient computation. Use 0 to disable.
    band_iters:
        Band thickness in voxels (morphological iterations).
    summary:
        Which single-number summary to return as `sharpness`.
        One of: "median", "mean", "p90", "p95", "max", "trimmed_mean".

    Returns
    -------
    Dict[str, float]
        Contains descriptive stats plus a single scalar `sharpness`.
    """
    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError("image and mask must be 3D arrays")

    if len(spacing) != 3:
        raise ValueError("spacing must be a length-3 tuple (sx, sy, sz)")
    if any(float(s) <= 0 for s in spacing):
        raise ValueError(f"spacing must be positive, got {spacing}")

    image_f = image.astype(np.float32, copy=False)
    mask_b = mask.astype(bool, copy=False)

    if band_iters < 1:
        raise ValueError("band_iters must be >= 1")

    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity

    # Inner boundary band: mask \ erode(mask)
    eroded = binary_erosion(mask_b, structure=structure,
                            iterations=band_iters)
    inner_band = mask_b & (~eroded)

    # Outer neighbor band: dilate(mask) \ mask
    dilated = binary_dilation(mask_b, structure=structure,
                              iterations=band_iters)
    outer_band = dilated & (~mask_b)

    band = inner_band | outer_band
    if int(band.sum()) == 0:
        raise ValueError("Band is empty. Check mask or band_iters.")

    # Gaussian smoothing: convert millimeters -> voxel sigmas per axis.
    if sigma_mm and float(sigma_mm) > 0:
        sigmas = tuple(float(sigma_mm) / float(sp) for sp in spacing)
        image_f = gaussian_filter(image_f, sigma=sigmas)

    # Gradients along (X, Y, Z) with spacing normalization.
    gx = sobel(image_f, axis=0) / float(spacing[0])
    gy = sobel(image_f, axis=1) / float(spacing[1])
    gz = sobel(image_f, axis=2) / float(spacing[2])

    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    values = grad_mag[band]

    if values.size == 0:
        raise ValueError("No gradient values found in band.")

    p10 = float(np.percentile(values, 10))
    p90 = float(np.percentile(values, 90))
    p95 = float(np.percentile(values, 95))

    # A single robust scalar for downstream grouping/comparison.
    if summary == "median":
        sharpness = float(np.median(values))
    elif summary == "mean":
        sharpness = float(values.mean())
    elif summary == "p90":
        sharpness = p90
    elif summary == "p95":
        sharpness = p95
    elif summary == "max":
        sharpness = float(values.max())
    elif summary == "trimmed_mean":
        # Mean of the upper tail: focuses on strongest edge voxels.
        sharpness = float(values[values >= p90].mean())
    else:
        raise ValueError(
            "summary must be one of: 'median', 'mean', 'p90', 'p95', "
            "'max', 'trimmed_mean'"
        )

    stats = {
        "n_band_voxels": int(values.size),
        "sharpness": sharpness,
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "p10": p10,
        "p90": p90,
        "p95": p95,
        "max": float(values.max()),
    }
    return stats


# Helper: scalar-only wrapper
def boundary_sharpness_scalar(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma_mm: float = 1.0,
    band_iters: int = 1,
    summary: str = "median",
) -> float:
    """Convenience wrapper that returns only the scalar sharpness."""
    stats = boundary_band_gradient_sharpness(
        image=image,
        mask=lesion_mask,
        spacing=spacing,
        sigma_mm=sigma_mm,
        band_iters=band_iters,
        summary=summary,
    )
    return float(stats["sharpness"])


def visualize_boundary_sharpness(
    image: np.ndarray,
    lesion_mask: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    sigma_mm: float = 1.0,
    band_iters: int = 1,
    summary: str = "median",
    margin: int = 3,
    show_full_context: bool = False,
    alpha: float = 0.55,
    vmax_percentile: float = 99.0,
    show_hist: bool = True,
) -> None:
    """Visualize lesion boundary sharpness as a band-limited heatmap.

    This visualization matches `boundary_band_gradient_sharpness`:
    - builds a boundary band (inside + outside)
    - computes gradient magnitude (optionally after mm-based smoothing)
    - overlays gradient magnitude in the band as a heatmap
    - draws the lesion contour in red and band contour in yellow

    Notes
    -----
    - Arrays are assumed to be in (X, Y, Z) order.
    - `lesion_mask` should contain only ONE lesion.
    """
    if image.shape != lesion_mask.shape:
        raise ValueError(
            "image and lesion_mask must have the same shape, "
            f"got {image.shape} vs {lesion_mask.shape}"
        )

    lesion_b = lesion_mask.astype(bool, copy=False)
    if int(lesion_b.sum()) == 0:
        raise ValueError("lesion_mask is empty")

    # Compute scalar stats (keeps metric + viz consistent).
    stats = boundary_band_gradient_sharpness(
        image=image,
        mask=lesion_b,
        spacing=spacing,
        sigma_mm=sigma_mm,
        band_iters=band_iters,
        summary=summary,
    )

    # Recompute band + grad magnitude for visualization.
    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    eroded = binary_erosion(lesion_b, structure=structure,
                            iterations=band_iters)
    inner_band = lesion_b & (~eroded)
    dilated = binary_dilation(lesion_b, structure=structure,
                              iterations=band_iters)
    outer_band = dilated & (~lesion_b)
    band = inner_band | outer_band

    image_f = image.astype(np.float32, copy=False)
    if sigma_mm and float(sigma_mm) > 0:
        sigmas = tuple(float(sigma_mm) / float(sp) for sp in spacing)
        image_f = gaussian_filter(image_f, sigma=sigmas)

    gx = sobel(image_f, axis=0) / float(spacing[0])
    gy = sobel(image_f, axis=1) / float(spacing[1])
    gz = sobel(image_f, axis=2) / float(spacing[2])
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)

    # Crop around lesion for readability.
    bbox = _get_bbox(lesion_b.astype(np.uint8), margin=margin)
    if bbox is None:
        raise ValueError("Could not compute bounding box for lesion")

    img_crop = image_f[bbox]
    msk_crop = lesion_b[bbox]
    band_crop = band[bbox]
    grad_crop = grad_mag[bbox]

    # Optional full-FOV axial context for ID sanity checks.
    if show_full_context:
        coords = np.where(lesion_b)
        zc = int(np.round(coords[2].mean()))
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(image[:, :, zc].T, cmap="gray", origin="lower")
        ax.contour(lesion_b[:, :, zc].T, colors="red", linewidths=1)
        ax.set_title(f"Full FOV axial (z={zc}, axis=2)")
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    # Choose consistent orthogonal slices through lesion centroid.
    coords_c = np.where(msk_crop)
    center = np.array(coords_c).mean(axis=1).astype(int)
    x, y, z = int(center[0]), int(center[1]), int(center[2])

    views = [
        (img_crop[x, :, :], msk_crop[x, :, :], band_crop[x, :, :],
         grad_crop[x, :, :], "Sagittal (x, axis=0)"),
        (img_crop[:, y, :], msk_crop[:, y, :], band_crop[:, y, :],
         grad_crop[:, y, :], "Coronal (y, axis=1)"),
        (img_crop[:, :, z], msk_crop[:, :, z], band_crop[:, :, z],
         grad_crop[:, :, z], "Axial (z, axis=2)"),
    ]

    # Scale heatmap for readability.
    band_vals = grad_crop[band_crop]
    if band_vals.size == 0:
        raise ValueError("Band is empty after cropping")

    vmax = float(np.percentile(band_vals, vmax_percentile))
    if vmax <= 0:
        vmax = float(band_vals.max())

    title = (
        f"Boundary sharpness ({summary}) = {stats['sharpness']:.3f} | "
        f"n_band={stats['n_band_voxels']} | sigma_mm={sigma_mm} | "
        f"band_iters={band_iters}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(title)

    for ax, (img2d, msk2d, band2d, g2d, name) in zip(axes, views):
        ax.imshow(img2d.T, cmap="gray", origin="lower")

        # Heatmap only in the band (else transparent).
        heat = np.full_like(g2d, np.nan, dtype=np.float32)
        heat[band2d] = g2d[band2d].astype(np.float32, copy=False)
        ax.imshow(
            heat.T,
            cmap="magma",
            origin="lower",
            alpha=alpha,
            vmin=0.0,
            vmax=vmax,
        )

        if np.any(msk2d):
            ax.contour(msk2d.T, colors="red", linewidths=1)
        if np.any(band2d):
            ax.contour(band2d.T, colors="yellow", linewidths=1)

        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    if show_hist:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        ax.hist(band_vals.astype(np.float32), bins=40)
        ax.set_title("Gradient magnitude in boundary band")
        ax.set_xlabel("|∇I| (a.u.)")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.show()


# -------------------------------------------
# End-to-end lesion table + CSV export
# -------------------------------------------

def _load_nifti_array_and_spacing(
    path: str | Path,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a NIfTI file and return (data, spacing_mm).

    Notes
    -----
    - Requires nibabel. If nibabel is not installed, this raises an error.
    - Returns arrays in the file's stored axis order. This code assumes your
      volumes are already aligned to the (X, Y, Z) convention used elsewhere
      in this module.
    """
    if nib is None:
        raise ImportError(
            "nibabel is required to load NIfTI files. Install with: "
            "pip install nibabel"
        )

    img = nib.load(str(path))
    data = np.asarray(img.get_fdata())

    zooms = img.header.get_zooms()
    if len(zooms) < 3:
        raise ValueError(
            f"Expected at least 3 zoom values, got {zooms} for {path}"
        )

    spacing_mm = (float(zooms[0]), float(zooms[1]), float(zooms[2]))
    return data, spacing_mm


def lesion_metrics_to_csv(
    image_path: str | Path,
    mask_path: str | Path,
    out_csv: str | Path,
    brain_mask_path: str | Path | None = None,
    *,
    connectivity: int = 3,
    ring_mm: float = 3.0,
    cnr_eps: float = 1e-8,
    sigma_mm: float = 1.0,
    band_iters: int = 1,
    sharpness_summary: str = "median",
    normalize: bool = True,
) -> list[dict[str, float]]:
    """Compute per-lesion metrics (volume, contrast, sharpness) and write a CSV.

    Steps
    -----
    1) Load image + lesion mask (NIfTI)
    2) Connected components on the binary lesion mask -> lesion IDs (1..N)
    3) For each lesion:
       - volume (mm^3)
       - local robust contrast (LCNR) using a perilesional ring
       - boundary sharpness via gradient magnitude in a thin boundary band

    CSV columns
    -----------
    lesion_id, volume_mm3, contrast, sharpness

    Notes
    -----
    - Contrast/sharpness are computed on `image_norm` (brain/foreground z-score).
    - Small lesions or rings may return NaN for contrast (see `local_robust_cnr`).
    """
    image, spacing_mm = _load_nifti_array_and_spacing(image_path)
    mask, _ = _load_nifti_array_and_spacing(mask_path)

    if image.shape != mask.shape:
        raise ValueError(
            "image and mask must have the same shape, "
            f"got {image.shape} vs {mask.shape}"
        )

    # Binary lesion mask (0 = background, 1 = lesion). This is what we label.
    lesion_mask_bin = (mask > 0).astype(np.uint8)

    if int(lesion_mask_bin.sum()) == 0:
        # No lesions -> write an empty CSV with header and return.
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["lesion_id", "volume_mm3", "contrast", "sharpness"],
            )
            writer.writeheader()
        return []

    # Brain/foreground mask used for normalization and for restricting the
    # perilesional background ring. If not provided, we derive it with fallbacks.
    brain_mask_bin: np.ndarray | None = None
    if brain_mask_path is not None:
        brain_mask_bin, _ = _load_nifti_array_and_spacing(brain_mask_path)
        if brain_mask_bin.shape != image.shape:
            raise ValueError(
                "brain_mask must have the same shape as image, "
                f"got {brain_mask_bin.shape} vs {image.shape}"
            )
        brain_mask_bin = (brain_mask_bin > 0).astype(np.uint8)

    # Normalize intensities for more comparable contrast/sharpness.
    # We normalize *within brain/foreground* so background does not dominate.
    if normalize:
        if brain_mask_bin is None:
            # Preferred heuristic (fast): background is exactly 0.
            brain_mask_bin = (image > 0).astype(np.uint8)

        # Rollbacks / safety checks (avoid empty mask -> NaNs).
        if int(brain_mask_bin.sum()) == 0:
            # Some pipelines can produce non-positive foreground; use non-zero.
            brain_mask_bin = (image != 0).astype(np.uint8)

        if int(brain_mask_bin.sum()) == 0:
            # As a last resort, normalize over full FOV (not ideal).
            brain_mask_bin = np.ones_like(lesion_mask_bin, dtype=np.uint8)

        image_norm = zscore_normalize(image, brain_mask_bin)
    else:
        image_norm = image.astype(np.float32, copy=False)

    lesion_label_map, num_lesions = connected_components(
        lesion_mask_bin,
        rank=3,
        connectivity=connectivity,
    )

    rows: list[dict[str, float]] = []

    for lesion_id in range(1, int(num_lesions) + 1):
        lesion_mask = (lesion_label_map == lesion_id)

        # Volume
        volume_mm3 = compute_lesion_volume(
            lesion_mask.astype(np.uint8),
            zoom=spacing_mm,
            unit="mm3",
        )

        # Local robust CNR (LCNR)
        ring_mask = lesion_background_ring(
            lesion_mask=lesion_mask.astype(np.uint8),
            ring_mm=ring_mm,
            spacing_mm=spacing_mm,
            brain_mask=brain_mask_bin,
        )
        contrast = local_robust_cnr(
            image_norm=image_norm,
            lesion_mask=lesion_mask,
            ring_mask=ring_mask,
            eps=cnr_eps,
        )

        # Boundary sharpness
        try:
            sharpness = boundary_sharpness_scalar(
                image=image_norm,
                lesion_mask=lesion_mask,
                spacing=spacing_mm,
                sigma_mm=sigma_mm,
                band_iters=band_iters,
                summary=sharpness_summary,
            )
        except ValueError:
            sharpness = float("nan")

        rows.append(
            {
                "lesion_id": float(lesion_id),
                "volume_mm3": float(volume_mm3),
                "contrast": float(contrast),
                "sharpness": float(sharpness),
            }
        )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["lesion_id", "volume_mm3", "contrast", "sharpness"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    return rows

