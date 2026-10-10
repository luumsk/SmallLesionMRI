"""Do small-lesion omissions change the periventricular lesion indicator?

The periventricular indicator (>= 1 lesion abutting the lateral ventricles) is
the leading predictor of CIS -> MS conversion in the CIStoCDMS study. This
script measures how segmentation errors of the four losses propagate into it.

For every test scan of MSLesSeg and 3D-MR-MS:
  1. Segment the lateral ventricles: the Harvard-Oxford lateral-ventricle map
     is warped from MNI space to the subject T2 (ANTs affine + SyN), dilated by
     3 mm, and intersected with the subject CSF mask (dark on FLAIR, bright on
     T2). Requires antspyx. The atlas files are created with nilearn:
       nilearn.datasets.load_mni152_template(resolution=1)       -> mni152_t1_1mm.nii.gz
       nilearn.datasets.fetch_atlas_harvard_oxford("sub-prob-1mm") -> ho_sub_prob_1mm.nii.gz
     and read from $PV_ATLAS_DIR (default: scripts/atlas).
  2. A lesion is periventricular if any of its voxels lies within D mm of the
     ventricle mask.
  3. The periventricular indicator of a scan is positive if it has >= 1
     periventricular lesion. It is computed from the reference mask and from
     the predicted mask of every loss and fold (raw and after the PP5 filter).

The location label is an automated, distance-based proxy of the radiological
definition; it is not radiologist-reviewed. Ventricle masks are cached as
<dataset>_<case>_vent.npy in $PV_VENT_CACHE (default: out_dir).

Outputs (pv_scans.csv, pv_lesions.csv, QC images) go to out_dir; the results
used in the dissertation are in metrics/periventricular/.
Summarize with periventricular_summary.py.

Usage: python periventricular_flip.py <out_dir> [--qc]
"""
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = "/Volumes/BACH2TB/Tunka"
DATASETS = {
    "MSLesSeg": ("nnUNet_raw/Dataset333_MSLesSeg", "SmallLesionMRI/MSLesSeg/final_checkpoints"),
    "3D-MR-MS": ("nnUNet_raw/Dataset666_LjubljanaMS", "SmallLesionMRI/LjubljanaMS/final_checkpoints"),
}
ATLAS_DIR = os.environ.get("PV_ATLAS_DIR", os.path.join(os.path.dirname(__file__), "atlas"))
MNI_TEMPLATE = os.path.join(ATLAS_DIR, "mni152_t1_1mm.nii.gz")  # nilearn load_mni152_template(resolution=1)
HO_SUB_PROB = os.path.join(ATLAS_DIR, "ho_sub_prob_1mm.nii.gz")  # nilearn fetch_atlas_harvard_oxford("sub-prob-1mm")
HO_LEFT_LV, HO_RIGHT_LV = 2, 13  # map indices of "Left/Right Lateral Ventricle" (labels 3 and 14 incl. background)
# brain masks released with 3D-MR-MS (the nnU-Net copies of its images are not skull-stripped)
BRAINMASK_3DMRMS = "/Volumes/BACH2TB/Datasets/3D-MR-MS/all"
LOSSES = {"CATMIL": "CATMIL", "DiceCE": "nnUNet", "Tversky": "Tversky", "FocalTversky": "FocalTversky"}
FOLDS = range(5)
DISTANCES_MM = [1, 2, 3, 5]
PP_MIN_VOX = 5
STRUCT6 = ndimage.generate_binary_structure(3, 1)


def load(path):
    img = nib.load(path)
    return np.asarray(img.dataobj).astype(np.float32), np.array(img.header.get_zooms()[:3], float)


def kmeans_centers(values, k=3, iters=30):
    """Sorted centers of k 1-D k-means clusters."""
    c = np.percentile(values, np.linspace(10, 90, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(values[:, None] - c[None]), 1)
        c = np.array([values[lab == i].mean() if np.any(lab == i) else c[i] for i in range(k)])
    return np.sort(c)


def csf_mask(flair, t2, brain):
    """CSF: dark on FLAIR and bright on T2 (MS lesions are bright on both)."""
    rng = np.random.default_rng(0)
    idx = np.flatnonzero(brain)
    sample = rng.choice(idx, min(200000, idx.size), replace=False)
    cf = kmeans_centers(flair.ravel()[sample])
    ct = kmeans_centers(t2.ravel()[sample])
    return brain & (flair < (cf[0] + cf[1]) / 2) & (t2 > (ct[1] + ct[2]) / 2)


def ventricle_prior(t2, brain, ref_path):
    """Harvard-Oxford lateral-ventricle probability warped to the subject.

    The MNI152 template is registered to the brain-masked T2 of the subject
    (affine + SyN, Mattes mutual information, on a 1 mm copy), and the summed
    left and right lateral-ventricle probability maps are warped with the same
    transform onto the subject grid.
    """
    import ants
    ref = ants.image_read(ref_path)  # subject grid (any channel file)
    fixed = ants.from_numpy((t2 * brain).astype(np.float32), origin=ref.origin,
                            spacing=ref.spacing, direction=ref.direction)
    fixed_1mm = ants.resample_image(fixed, (1, 1, 1), use_voxels=False, interp_type=0)
    moving = ants.image_read(MNI_TEMPLATE)
    reg = ants.registration(fixed=fixed_1mm, moving=moving, type_of_transform="SyN",
                            aff_metric="mattes", syn_metric="mattes", random_seed=0)
    ho = nib.load(HO_SUB_PROB)
    probs = np.asarray(ho.dataobj)[..., [HO_LEFT_LV, HO_RIGHT_LV]].sum(-1).astype(np.float32) / 100.0
    tmp = os.path.join(os.path.dirname(HO_SUB_PROB), "_lv_prob.nii.gz")
    nib.save(nib.Nifti1Image(probs, ho.affine), tmp)
    warped = ants.apply_transforms(fixed=fixed, moving=ants.image_read(tmp),
                                   transformlist=reg["fwdtransforms"], interpolator="linear")
    return warped.numpy()


def segment_lateral_ventricles(flair, t2, brain, zooms, ref_path):
    """Lateral ventricles: subject CSF inside the warped atlas ventricles (dilated by 3 mm).

    The periventricular region of the McDonald criteria refers to the lateral
    ventricles; the atlas prior excludes the fissure, sulci, cisterns and the
    third and fourth ventricles, and the subject CSF mask gives the boundary.
    """
    prior = ventricle_prior(t2, brain, ref_path) > 0.25
    near = ndimage.distance_transform_edt(~prior, sampling=zooms) <= 3
    vent = csf_mask(flair, t2, brain) & near
    lab, n = ndimage.label(vent, structure=STRUCT6)
    if n:
        vol = ndimage.sum_labels(vent, lab, np.arange(1, n + 1)) * np.prod(zooms)
        vent = np.isin(lab, np.flatnonzero(vol >= 50) + 1)  # drop isolated specks
    return vent


def brain_mask(dataset, case, flair):
    if dataset == "3D-MR-MS":
        m = load(f"{BRAINMASK_3DMRMS}/{case}/{case}_brainmask.nii.gz")[0] > 0
    else:
        m = flair > 0
    return ndimage.binary_fill_holes(m)


def periventricular_flags(mask, vent_dist):
    """Label 6-connected components of mask; return (labels, n, min distance per component)."""
    lab, n = ndimage.label(mask, structure=STRUCT6)
    if n == 0:
        return lab, n, np.array([]), np.array([])
    ids = np.arange(1, n + 1)
    dmin = ndimage.minimum(vent_dist, lab, ids)
    size = ndimage.sum_labels(mask, lab, ids)
    return lab, n, np.asarray(dmin), np.asarray(size)


def qc_image(path, t2, flair, vent, gt, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zs = np.flatnonzero(vent.any((0, 1)))
    if zs.size == 0:
        zs = np.array([t2.shape[2] // 2])
    picks = np.linspace(zs.min(), zs.max(), 5).round().astype(int)
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    for j, z in enumerate(picks):
        for i, img in enumerate((t2, flair)):
            ax[i, j].imshow(np.rot90(img[:, :, z]), cmap="gray")
            ax[i, j].contour(np.rot90(vent[:, :, z]), [0.5], colors="cyan", linewidths=0.8)
            ax[i, j].contour(np.rot90(gt[:, :, z]), [0.5], colors="red", linewidths=0.6)
            ax[i, j].set_axis_off()
            ax[i, j].set_title(f"z={z}", fontsize=8)
    fig.suptitle(title + "  (cyan: lateral ventricles, red: reference lesions; top T2, bottom FLAIR)")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def process_case(dataset, case, out_dir, qc):
    raw, pred_root = DATASETS[dataset]
    t2, zooms = load(f"{ROOT}/{raw}/imagesTs/{case}_0002.nii.gz")
    flair, _ = load(f"{ROOT}/{raw}/imagesTs/{case}_0003.nii.gz")
    gt = load(f"{ROOT}/{raw}/labelsTs/{case}.nii.gz")[0] > 0.5
    brain = brain_mask(dataset, case, flair)
    cache = os.path.join(os.environ.get("PV_VENT_CACHE", out_dir), f"{dataset}_{case}_vent.npy")
    if os.path.exists(cache):
        vent = np.load(cache)
    else:
        vent = segment_lateral_ventricles(flair, t2, brain, zooms, f"{ROOT}/{raw}/imagesTs/{case}_0003.nii.gz")
        np.save(os.path.join(out_dir, f"{dataset}_{case}_vent.npy"), vent)
    vent_dist = ndimage.distance_transform_edt(~vent, sampling=zooms)
    if qc:
        qc_image(f"{out_dir}/qc_{dataset}_{case}.png", t2, flair, vent, gt, f"{dataset} {case}")

    gt_lab, gt_n, gt_dmin, gt_size = periventricular_flags(gt, vent_dist)
    lesion_rows, scan_rows = [], []
    for loss, folder in LOSSES.items():
        for variant in ("raw", "pp5"):
            for fold in FOLDS:
                seg = load(f"{ROOT}/{pred_root}/{folder}/fold_{fold}/{case}.nii.gz")[0] > 0
                p_lab, p_n, p_dmin, p_size = periventricular_flags(seg, vent_dist)
                if variant == "pp5" and p_n:
                    small = np.flatnonzero(p_size < PP_MIN_VOX) + 1
                    seg = seg & ~np.isin(p_lab, small)
                    p_lab, p_n, p_dmin, p_size = periventricular_flags(seg, vent_dist)
                detected = ndimage.maximum(seg, gt_lab, np.arange(1, gt_n + 1)) > 0 if gt_n else np.array([])
                if variant == "raw":
                    for k in range(gt_n):
                        lesion_rows.append({
                            "dataset": dataset, "case": case, "loss": loss, "fold": fold,
                            "lesion_id": k + 1, "size_vox": int(gt_size[k]),
                            "dist_to_ventricle_mm": float(gt_dmin[k]), "detected": bool(detected[k]),
                        })
                for d in DISTANCES_MM:
                    pv_gt = gt_dmin <= d
                    scan_rows.append({
                        "dataset": dataset, "case": case, "loss": loss, "variant": variant,
                        "fold": fold, "d_mm": d,
                        "ventricle_ml": float(vent.sum() * np.prod(zooms) / 1000),
                        "gt_pv_lesions": int(pv_gt.sum()),
                        "gt_pv_small_lesions": int((pv_gt & (gt_size <= 150)).sum()),
                        "pv_lesions_missed": int((pv_gt & ~detected).sum()) if gt_n else 0,
                        "pv_small_missed": int((pv_gt & (gt_size <= 150) & ~detected).sum()) if gt_n else 0,
                        "indicator_gt": bool(pv_gt.any()),
                        # indicator from the prediction alone (includes false-positive components)
                        "indicator_pred": bool((p_dmin <= d).any()) if p_n else False,
                        # indicator from detected reference lesions only
                        "indicator_detected": bool((pv_gt & detected).any()) if gt_n else False,
                    })
    print(dataset, case, f"ventricles {vent.sum() * np.prod(zooms) / 1000:.1f} ml", f"lesions {gt_n}", flush=True)
    return lesion_rows, scan_rows


def main(out_dir, qc=False):
    os.makedirs(out_dir, exist_ok=True)
    lesion_rows, scan_rows = [], []
    for dataset, (raw, _) in DATASETS.items():
        cases = sorted(f[:-7] for f in os.listdir(f"{ROOT}/{raw}/labelsTs") if f.endswith(".nii.gz"))
        for case in cases:
            lr, sr = process_case(dataset, case, out_dir, qc)
            lesion_rows += lr
            scan_rows += sr
    pd.DataFrame(lesion_rows).to_csv(f"{out_dir}/pv_lesions.csv", index=False)
    pd.DataFrame(scan_rows).to_csv(f"{out_dir}/pv_scans.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1], "--qc" in sys.argv)
