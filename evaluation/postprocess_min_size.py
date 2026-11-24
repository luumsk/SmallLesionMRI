"""Remove small predicted components from nnU-Net masks (PP3, PP5, PP10, PP15).

Every 6-connected component with fewer than --min_voxels voxels is removed,
the same connectivity that multiscale_eval.py uses for lesion matching.
Masks are written with the input header, so they keep the input geometry.

Usage:
    python evaluation/postprocess_min_size.py --in_dir <pred>/CATMIL/fold_0 \
        --out_dir <pred>/CATMIL_pp5/fold_0 --min_voxels 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


def remove_small_components(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    """Keep only 6-connected components with at least min_voxels voxels."""
    struct = ndimage.generate_binary_structure(rank=3, connectivity=1)
    lab, n = ndimage.label(mask, structure=struct)
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel(), minlength=n + 1) >= min_voxels
    keep[0] = False
    return keep[lab]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in_dir", required=True, help="Folder with predicted masks (*.nii.gz).")
    p.add_argument("--out_dir", required=True, help="Output folder for the filtered masks.")
    p.add_argument("--min_voxels", type=int, default=5,
                   help="Components with fewer voxels are removed (paper: 5).")
    args = p.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    paths = sorted(in_dir.glob("*.nii.gz"))
    if not paths:
        raise FileNotFoundError(f"No *.nii.gz masks in {in_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in paths:
        img = nib.load(str(path))
        mask = np.asanyarray(img.dataobj) > 0.5
        kept = remove_small_components(mask, args.min_voxels)
        out = nib.Nifti1Image(kept.astype(np.uint8), img.affine, img.header)
        out.set_data_dtype(np.uint8)
        nib.save(out, str(out_dir / path.name))
        print(f"{path.name}: {int(mask.sum())} -> {int(kept.sum())} voxels")


if __name__ == "__main__":
    main()
