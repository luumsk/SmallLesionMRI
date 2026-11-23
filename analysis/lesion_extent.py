"""Largest linear measurement of every ground-truth lesion (MAGNIMS lesion size definition).

MAGNIMS recommends that lesions be "at least 3 mm in longest diameter" (Barkhof et al.,
Lancet Neurol 2025; earlier: >= 3 mm in at least one plane, Filippi et al., Lancet Neurol
2016). For each lesion this computes
  max_inplane_mm  the largest in-plane diameter over all axial, coronal and sagittal
                  slices through the lesion (what a reader would measure)
  feret_3d_mm     the largest 3D diameter (upper bound of the in-plane value)
Diameters are edge to edge: the distance between the two furthest voxel centres plus
one voxel (mean in-plane spacing), so a single voxel measures one voxel width.

Lesions are labelled exactly as in build_lesion_tables.py (6-connected components of
the label > 0.5), so lesion_id matches lesions.csv.

Usage: python lesion_extent.py <out_csv> [n_workers]
"""
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist

from build_lesion_tables import DATASETS, ROOT, label


def max_distance(points):
    """Largest pairwise distance, computed on the convex hull when it exists."""
    if len(points) < 2:
        return 0.0
    try:
        points = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):  # collinear / coplanar / too few points
        pass
    return float(pdist(points).max())


def lesion_extent(coords, spacing):
    mm = coords * spacing
    best = 0.0
    for axis in range(3):
        keep = [a for a in range(3) if a != axis]
        width = spacing[keep].mean()
        for s in np.unique(coords[:, axis]):
            sl = mm[coords[:, axis] == s][:, keep]
            best = max(best, max_distance(sl) + width)
    return best, max_distance(mm) + spacing.mean()


def process(job):
    dataset, case = job
    _, gt_dir = DATASETS[dataset]
    img = nib.load(f"{ROOT}/{gt_dir}/{case}.nii.gz")
    gt = np.asarray(img.dataobj) > 0.5
    spacing = np.array(img.header.get_zooms()[:3], dtype=float)
    gt_lab, gt_n = label(gt)
    objects = ndimage.find_objects(gt_lab)
    rows = []
    for i, sl in enumerate(objects, start=1):
        offset = np.array([s.start for s in sl])
        coords = np.argwhere(gt_lab[sl] == i) + offset
        inplane, feret = lesion_extent(coords, spacing)
        rows.append(dict(dataset=dataset, case=case, lesion_id=i, size_vox=len(coords),
                         max_inplane_mm=inplane, feret_3d_mm=feret))
    print(dataset, case, gt_n, flush=True)
    return pd.DataFrame(rows)


def jobs():
    for dataset, (_, gt_dir) in DATASETS.items():
        for f in sorted(os.listdir(f"{ROOT}/{gt_dir}")):
            if f.endswith(".nii.gz"):
                yield dataset, f[:-7]


def main(out_csv, workers=4):
    with Pool(workers) as pool:
        results = pool.map(process, list(jobs()), chunksize=1)
    pd.concat(results).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
