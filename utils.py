from typing import Dict, Tuple
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt

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
def connected_components(mask):
    """
    Computes connected components in a 3D binary mask.

    Parameters:
    mask (numpy.ndarray): A 3D binary numpy array where non-zero values indicate the presence of a lesion.

    Returns:
    tuple: A tuple containing:
        - labeled_mask (numpy.ndarray): A 3D array where each connected component is assigned a unique integer label.
        - num_components (int): The number of connected components found in the mask.
    """
    structure = generate_binary_structure(rank=3, connectivity=3)  # 26-connectivity
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
    """
    rad = [
        max(1, int(np.ceil(ring_mm / float(sp))))
        for sp in spacing_mm
    ]

    struct = ndi.generate_binary_structure(rank=3, connectivity=1)
    dilated = ndi.binary_dilation(
        lesion_mask,
        structure=struct,
        iterations=max(rad),
    )

    ring = np.logical_and(dilated, np.logical_not(lesion_mask))

    if brain_mask is not None:
        ring = np.logical_and(ring, brain_mask)

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
                f"Lesion {lid} | Contrast = {val:.3f} | Volume={int(mask_crop.sum())}"
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
    sigma: float = 1.0,
    band_iters: int = 1,
) -> Dict[str, float]:
    """
    Sharpness = gradient magnitude sampled in a thin band around lesion
    (inside + outside). Connectivity is 26 via 3x3x3 structuring element.

    band_iters controls band thickness in voxels.
    """
    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError("image and mask must be 3D arrays")

    image = image.astype(np.float32)
    mask = mask.astype(bool)

    if band_iters < 1:
        raise ValueError("band_iters must be >= 1")

    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity

    # Inner boundary band: mask \ erode(mask)
    eroded = binary_erosion(mask, structure=structure, iterations=band_iters)
    inner_band = mask & (~eroded)

    # Outer neighbor band: dilate(mask) \ mask
    dilated = binary_dilation(mask, structure=structure, iterations=band_iters)
    outer_band = dilated & (~mask)

    band = inner_band | outer_band
    if band.sum() == 0:
        raise ValueError("Band is empty. Check mask or band_iters.")

    if sigma > 0:
        image = gaussian_filter(image, sigma=sigma)

    gz = sobel(image, axis=0) / float(spacing[0])
    gy = sobel(image, axis=1) / float(spacing[1])
    gx = sobel(image, axis=2) / float(spacing[2])

    grad_mag = np.sqrt(gz * gz + gy * gy + gx * gx)
    values = grad_mag[band]

    stats = {
        "n_band_voxels": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }
    return stats

