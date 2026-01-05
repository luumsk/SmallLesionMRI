import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label, generate_binary_structure

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