from typing import Dict, Tuple

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
    """Displays a slice of the MRI image alongside its corresponding segmentation mask."""
    
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