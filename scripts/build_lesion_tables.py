"""Per-lesion and per-component tables for the four losses on both test sets.

For every dataset, loss, fold model and test case this writes
  lesions.csv     one row per ground-truth lesion: size, detection, coverage,
                  max foreground probability inside the lesion, and, for each
                  probability threshold t, the size (voxels) of the largest
                  predicted component at p > t that overlaps the lesion
                  (0 = no overlap). A lesion is detected at threshold t with
                  minimum component size s iff that size is >= s.
  components.csv  one row per predicted component at each threshold t:
                  size (voxels) and whether it overlaps the ground truth.

Detection follows multiscale_eval.py: 6-connected components, any-voxel overlap.
The nnU-Net segmentation equals p > 0.5, so t = 0.5 reproduces the saved masks.

Usage: python build_lesion_tables.py <out_dir> [n_workers]
"""
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = "/Volumes/BACH2TB/Tunka"
DATASETS = {
    "MSLesSeg": ("SmallLesionMRI/MSLesSeg/final_checkpoints", "nnUNet_raw/Dataset333_MSLesSeg/labelsTs"),
    "3D-MR-MS": ("SmallLesionMRI/LjubljanaMS/final_checkpoints", "nnUNet_raw/Dataset666_LjubljanaMS/labelsTs"),
}
MODELS = {"CATMIL": "CATMIL", "DiceCE": "nnUNet", "Tversky": "Tversky", "FocalTversky": "FocalTversky"}
FOLDS = range(5)
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
STRUCT = ndimage.generate_binary_structure(3, 1)


def label(mask):
    lab, n = ndimage.label(mask, structure=STRUCT)
    return lab, n


def process(job):
    dataset, model, fold, case = job
    pred_dir, gt_dir = DATASETS[dataset]
    gt_img = nib.load(f"{ROOT}/{gt_dir}/{case}.nii.gz")
    gt = np.asarray(gt_img.dataobj) > 0.5
    voxel_mm3 = float(np.prod(gt_img.header.get_zooms()[:3]))
    d = f"{ROOT}/{pred_dir}/{MODELS[model]}/fold_{fold}"
    # nnU-Net stores probabilities in SimpleITK (z, y, x) order
    prob = np.load(f"{d}/{case}.npz")["probabilities"][1].transpose(2, 1, 0)
    seg = np.asarray(nib.load(f"{d}/{case}.nii.gz").dataobj) > 0
    assert prob.shape == gt.shape == seg.shape

    gt_lab, gt_n = label(gt)
    ids = np.arange(1, gt_n + 1)
    sizes = ndimage.sum_labels(gt, gt_lab, ids)
    covered = ndimage.sum_labels(seg, gt_lab, ids)
    max_prob = ndimage.maximum(prob, gt_lab, ids)
    centroids = ndimage.center_of_mass(gt, gt_lab, ids)
    lesions = pd.DataFrame({
        "dataset": dataset, "model": model, "fold": fold, "case": case, "lesion_id": ids,
        "size_vox": sizes.astype(int), "size_mm3": sizes * voxel_mm3,
        "detected": covered > 0, "coverage": covered / sizes, "max_prob": max_prob,
        "cx": [c[0] for c in centroids], "cy": [c[1] for c in centroids], "cz": [c[2] for c in centroids],
    })

    comp_rows = []
    gt_idx = gt_lab[gt]
    for t in THRESHOLDS:
        pr_lab, pr_n = label(prob > t)
        if pr_n == 0:
            lesions[f"hit_size_t{t}"] = 0
            continue
        pr_sizes = np.bincount(pr_lab.ravel(), minlength=pr_n + 1)
        pr_idx = pr_lab[gt]
        pairs = np.unique(np.stack([gt_idx, pr_idx], 1)[pr_idx > 0], axis=0)
        hit = np.zeros(gt_n + 1, dtype=int)
        np.maximum.at(hit, pairs[:, 0], pr_sizes[pairs[:, 1]])
        lesions[f"hit_size_t{t}"] = hit[1:]
        matched = np.zeros(pr_n + 1, dtype=bool)
        matched[pairs[:, 1]] = True
        comp_rows.append(pd.DataFrame({
            "dataset": dataset, "model": model, "fold": fold, "case": case, "threshold": t,
            "size_vox": pr_sizes[1:], "matched": matched[1:],
        }))
    comps = pd.concat(comp_rows) if comp_rows else pd.DataFrame()
    print(dataset, model, fold, case, gt_n, flush=True)
    return lesions, comps


def jobs():
    for dataset, (pred_dir, gt_dir) in DATASETS.items():
        cases = sorted(f[:-7] for f in os.listdir(f"{ROOT}/{gt_dir}") if f.endswith(".nii.gz"))
        for model in MODELS:
            for fold in FOLDS:
                for case in cases:
                    yield dataset, model, fold, case


def main(out_dir, workers=4):
    os.makedirs(out_dir, exist_ok=True)
    with Pool(workers) as pool:
        results = pool.map(process, list(jobs()), chunksize=1)
    pd.concat([r[0] for r in results]).to_csv(f"{out_dir}/lesions.csv", index=False)
    pd.concat([r[1] for r in results]).to_csv(f"{out_dir}/components.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4)
