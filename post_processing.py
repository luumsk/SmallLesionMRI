

"""Post-process segmentation masks by removing small components.

This script scans an input folder for NIfTI prediction masks, removes
connected components smaller than a user-defined voxel threshold, and
writes the filtered masks to an output folder while preserving affine and
header metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from scipy import ndimage


VALID_SUFFIXES = (".nii", ".nii.gz")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Remove connected components smaller than a voxel threshold "
            "from segmentation masks in a folder."
        )
    )
    parser.add_argument(
        "-i",
        dest="input_dir",
        type=Path,
        help="Folder containing predicted segmentation masks (.nii/.nii.gz).",
    )
    parser.add_argument(
        "-o",
        dest="output_dir",
        type=Path,
        help="Folder to save filtered masks.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        required=True,
        help="Minimum component size in voxels to keep.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(1, 2, 3),
        default=1,
        help=(
            "Connectivity for 3D connected components: 1=6-neighborhood, "
            "2=18-neighborhood, 3=26-neighborhood. Default: 1."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for masks inside subfolders.",
    )
    return parser.parse_args()


def find_mask_files(input_dir: Path, recursive: bool) -> list[Path]:
    """Find NIfTI mask files in a directory."""
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.name.endswith(VALID_SUFFIXES)
    ]
    return sorted(files)


def build_structure(connectivity: int) -> np.ndarray:
    """Build a binary structure for connected component analysis."""
    return ndimage.generate_binary_structure(rank=3, connectivity=connectivity)


def remove_small_components(
    mask: np.ndarray,
    min_size: int,
    structure: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Remove connected components smaller than min_size.

    Parameters
    ----------
    mask:
        Input binary mask.
    min_size:
        Minimum number of voxels required to keep a component.
    structure:
        Connectivity structure used by connected component analysis.

    Returns
    -------
    filtered_mask:
        Binary mask after removing small components.
    num_original:
        Number of connected components before filtering.
    num_removed:
        Number of connected components removed.
    """
    labeled, num_features = ndimage.label(mask, structure=structure)
    if num_features == 0:
        return mask.astype(np.uint8), 0, 0

    component_sizes = np.bincount(labeled.ravel())
    keep_labels = np.where(component_sizes >= min_size)[0]
    keep_labels = keep_labels[keep_labels != 0]

    filtered_mask = np.isin(labeled, keep_labels)
    num_removed = int(num_features - len(keep_labels))
    return filtered_mask.astype(np.uint8), int(num_features), num_removed


def load_mask(mask_path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Load a NIfTI mask and convert it to a binary array."""
    image = nib.load(str(mask_path))
    data = image.get_fdata()
    binary_mask = (data > 0).astype(np.uint8)
    return image, binary_mask


def save_mask(
    output_path: Path,
    filtered_mask: np.ndarray,
    reference_img: nib.Nifti1Image,
) -> None:
    """Save the filtered mask using reference affine and header."""
    header = reference_img.header.copy()
    header.set_data_dtype(np.uint8)
    output_img = nib.Nifti1Image(
        filtered_mask,
        affine=reference_img.affine,
        header=header,
    )
    nib.save(output_img, str(output_path))


def ensure_output_path(
    input_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> Path:
    """Create the output path while preserving relative structure."""
    relative_path = input_path.relative_to(input_dir)
    output_path = output_dir / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def process_masks(
    mask_paths: Iterable[Path],
    input_dir: Path,
    output_dir: Path,
    min_size: int,
    connectivity: int,
) -> None:
    """Process and save all masks."""
    structure = build_structure(connectivity)

    total_files = 0
    total_components = 0
    total_removed = 0

    for mask_path in mask_paths:
        reference_img, binary_mask = load_mask(mask_path)
        filtered_mask, num_components, num_removed = remove_small_components(
            mask=binary_mask,
            min_size=min_size,
            structure=structure,
        )

        output_path = ensure_output_path(mask_path, input_dir, output_dir)
        save_mask(output_path, filtered_mask, reference_img)

        total_files += 1
        total_components += num_components
        total_removed += num_removed

        print(
            f"Processed: {mask_path.name} | "
            f"components={num_components} | removed={num_removed}"
        )

    print("\nDone.")
    print(f"Files processed: {total_files}")
    print(f"Total components: {total_components}")
    print(f"Total removed: {total_removed}")


def main() -> None:
    """Run the command-line script."""
    args = parse_args()

    if args.min_size < 1:
        raise ValueError("--min-size must be at least 1.")

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {args.input_dir}"
        )

    if not args.input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {args.input_dir}"
        )

    mask_paths = find_mask_files(args.input_dir, args.recursive)
    if not mask_paths:
        raise FileNotFoundError(
            f"No NIfTI files found in: {args.input_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    process_masks(
        mask_paths=mask_paths,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        min_size=args.min_size,
        connectivity=args.connectivity,
    )


if __name__ == "__main__":
    main()