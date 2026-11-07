"""Chapter 3 on all 53 MSLesSeg patients: out-of-fold (cross-validation) evaluation.

The Ch3 test set has 12 scans of 6 patients. nnU-Net also stored, for every one
of the 81 training scans (47 other patients), the prediction of the fold model
that did not see that scan during training (fold_k/validation/). No model is
trained here. The CATMIL weights were fixed before the experiments
(sections/small_lesions/method.tex), so the validation folds were not used to
choose them.

Metrics per scan, from the saved masks (p > 0.5), exactly as
scripts/build_lesion_tables.py + scripts/recall_bootstrap.py define them:
6-connected components; a reference lesion is detected if any predicted voxel
overlaps it; small = <= 150 voxels; lesion F1 from lesion-wise recall and
component-wise precision; with --pp5-style variants, predicted components
< 5 voxels are removed first (a lesion stays detected only if an overlapping
component of >= 5 voxels exists). Also DSC per scan.

Step 0 (check): the same code on the stored test-set fold predictions must
reproduce the dissertation (CATMIL small recall 0.8730, DiceCE 0.7956).
Step 1: out-of-fold metrics on the 81 training scans; each scan is predicted
once. Mean over scans; patient-level bootstrap (2000 resamples of the 47
patients, scans of a patient kept together) of CATMIL - each other model.
Step 2: the same on all 93 scans of 53 patients (out-of-fold + test; for the
test scans the per-scan value is the mean over the 5 fold models).

Usage: python oof_evaluation.py [n_workers]
Outputs: ../results/oof_per_scan.csv, oof_summary.csv, oof_bootstrap.csv, oof_testcheck.csv
"""
import os
import sys
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = "/Volumes/BACH2TB/Tunka"
RES = f"{ROOT}/nnUNet_results/Dataset333_MSLesSeg"
TEST_PRED = f"{ROOT}/SmallLesionMRI/MSLesSeg/final_checkpoints"
GT_TR = f"{ROOT}/nnUNet_raw/Dataset333_MSLesSeg/labelsTr"
GT_TS = f"{ROOT}/nnUNet_raw/Dataset333_MSLesSeg/labelsTs"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
# display name -> (trainer folder in nnUNet_results, folder in final_checkpoints)
MODELS = {
    "CATMIL": ("nnUNetTrainerCATMIL", "CATMIL"),
    "DiceCE": ("nnUNetTrainer", "nnUNet"),
    "Tversky": ("nnUNetTrainerTversky", "Tversky"),
    "FocalTversky": ("nnUNetTrainerFocalTversky", "FocalTversky"),
    "SegResNet (DiceCE)": ("nnUNetTrainerSegResNet", "SegResNet"),
    "SwinUNETR (DiceCE)": ("nnUNetTrainerSwinUNETR", "SwinUNETR"),
    "UNETR (DiceCE)": ("nnUNetTrainerUNETR", "UNETR"),
    "UMambaBot (DiceCE)": ("nnUNetTrainerUMambaBot", "UMambaBot"),
}
STRUCT = ndimage.generate_binary_structure(3, 1)
SMALL_VOX, EPS = 150, 1e-8
METRICS = ["small_recall", "fn_count", "lesion_f1", "small_recall_pp5", "fn_count_pp5", "lesion_f1_pp5", "dice"]


def scan_metrics(gt, pr):
    g_lab, g_n = ndimage.label(gt, structure=STRUCT)
    p_lab, p_n = ndimage.label(pr, structure=STRUCT)
    g_size = np.bincount(g_lab.ravel(), minlength=g_n + 1)[1:]
    p_size = np.bincount(p_lab.ravel(), minlength=p_n + 1)
    # largest predicted component overlapping each reference lesion; matched predicted components
    pairs = np.unique(np.stack([g_lab[gt], p_lab[gt]], 1), axis=0)
    pairs = pairs[pairs[:, 1] > 0]
    hit = np.zeros(g_n + 1, dtype=int)
    np.maximum.at(hit, pairs[:, 0], p_size[pairs[:, 1]])
    hit = hit[1:]
    matched = np.zeros(p_n + 1, dtype=bool)
    matched[pairs[:, 1]] = True
    matched, p_size = matched[1:], p_size[1:]
    out = {"n_lesions": int(g_n), "dice": float(2 * (gt & pr).sum() / (gt.sum() + pr.sum() + EPS))}
    for sfx, k in (("", 0), ("_pp5", 5)):
        det = hit >= max(k, 1)
        keep = p_size >= max(k, 1)
        small = g_size <= SMALL_VOX
        rec = det.sum() / (g_n + EPS)
        prec = matched[keep].sum() / (keep.sum() + EPS)
        out[f"small_recall{sfx}"] = float((det & small).sum() / (small.sum() + EPS))
        out[f"fn_count{sfx}"] = int(g_n - det.sum())
        out[f"lesion_f1{sfx}"] = float(2 * prec * rec / (prec + rec + EPS))
    return out


def job(a):
    model, fold, case, path, gt_path, split = a
    gt = np.asarray(nib.load(gt_path).dataobj).astype(np.float32) > 0.5
    pr = np.asarray(nib.load(path).dataobj).astype(np.float32) > 0.5
    assert gt.shape == pr.shape, (path, gt.shape, pr.shape)
    return {"model": model, "fold": fold, "case": case, "split": split, **scan_metrics(gt, pr)}


def jobs():
    for m, (trainer, final) in MODELS.items():
        for k in range(5):
            vdir = f"{RES}/{trainer}__nnUNetPlans__3d_fullres/fold_{k}/validation"
            for f in sorted(os.listdir(vdir)) if os.path.isdir(vdir) else []:
                if f.endswith(".nii.gz"):
                    yield m, k, f[:-7], f"{vdir}/{f}", f"{GT_TR}/{f}", "oof"
            tdir = f"{TEST_PRED}/{final}/fold_{k}"
            for f in sorted(os.listdir(tdir)) if os.path.isdir(tdir) else []:
                if f.endswith(".nii.gz"):
                    yield m, k, f[:-7], f"{tdir}/{f}", f"{GT_TS}/{f}", "test"


def bootstrap(per_scan, models, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    pats = sorted(per_scan.patient.unique())
    by_p = {p: per_scan[per_scan.patient == p] for p in pats}
    rows = []
    boots = []
    for _ in range(n):
        d = pd.concat([by_p[p] for p in rng.choice(pats, len(pats))])
        boots.append(d.groupby("model")[METRICS].mean())
    point = per_scan.groupby("model")[METRICS].mean()
    for m in models:
        if m == "CATMIL":
            continue
        for met in METRICS:
            diff = np.array([b.loc["CATMIL", met] - b.loc[m, met] for b in boots])
            rows.append({"comparison": f"CATMIL - {m}", "metric": met,
                         "diff": point.loc["CATMIL", met] - point.loc[m, met],
                         "ci_lo": np.quantile(diff, 0.025), "ci_hi": np.quantile(diff, 0.975),
                         "share_patients_catmil_better": np.nan})
    return pd.DataFrame(rows)


def main(workers=4):
    os.makedirs(OUT, exist_ok=True)
    with Pool(workers) as p:
        df = pd.DataFrame(p.map(job, list(jobs()), chunksize=1))
    df["patient"] = df.case.str.split("_").str[0]
    df.to_csv(os.path.join(OUT, "oof_per_scan.csv"), index=False)
    models = list(MODELS)

    # Step 0: test-set check (mean over scans per fold, then over folds)
    test = df[df.split == "test"]
    chk = test.groupby(["model", "fold"])[METRICS].mean().groupby("model").mean().loc[
        [m for m in models if m in test.model.unique()]]
    chk.to_csv(os.path.join(OUT, "oof_testcheck.csv"))
    print("TEST-SET CHECK (expect CATMIL small_recall 0.8730, DiceCE 0.7956; PP5 CATMIL 0.8553)")
    print(chk[["small_recall", "fn_count", "lesion_f1", "small_recall_pp5", "lesion_f1_pp5", "dice"]].round(4).to_string())

    # Step 1: out-of-fold, 81 scans / 47 patients
    oof = df[df.split == "oof"]
    n_scans = oof.groupby("model").case.nunique()
    print("\nOOF scans per model:", n_scans.to_dict())
    summ = oof.groupby("model")[METRICS + ["n_lesions"]].mean().loc[models]
    summ.insert(0, "n_scans", n_scans.loc[models])
    summ.insert(1, "n_patients", oof.groupby("model").patient.nunique().loc[models])
    # Step 2: all 93 scans / 53 patients (test scans: mean over fold models)
    test_scan = test.groupby(["model", "case", "patient"])[METRICS + ["n_lesions"]].mean().reset_index()
    allv = pd.concat([oof[["model", "case", "patient"] + METRICS + ["n_lesions"]], test_scan])
    summ_all = allv.groupby("model")[METRICS].mean().loc[[m for m in models if m in test_scan.model.unique()]]
    summ_all.columns = [f"{c}_all93" for c in summ_all.columns]
    summ = summ.join(summ_all)
    summ.to_csv(os.path.join(OUT, "oof_summary.csv"))
    print("\nOUT-OF-FOLD, 81 scans of 47 patients (and *_all93 = 93 scans of 53 patients)")
    print(summ.round(4).to_string())

    bt = bootstrap(oof, models)
    bt["set"] = "oof_47_patients"
    bt2 = bootstrap(allv[allv.model.isin(summ_all.index)], [m for m in models if m in summ_all.index])
    bt2["set"] = "all_53_patients"
    bt = pd.concat([bt, bt2])
    # per-patient direction for small recall (CATMIL vs DiceCE)
    pp = oof.groupby(["model", "patient"]).small_recall.mean().unstack(0)
    better = float((pp["CATMIL"] > pp["DiceCE"]).mean())
    worse = float((pp["CATMIL"] < pp["DiceCE"]).mean())
    print(f"\nOOF per patient small recall CATMIL vs DiceCE: better {better:.3f}, worse {worse:.3f}, n={len(pp)}")
    bt.to_csv(os.path.join(OUT, "oof_bootstrap.csv"), index=False)
    print(bt[bt.metric.isin(["small_recall", "fn_count", "lesion_f1", "lesion_f1_pp5", "dice"])].round(4).to_string(index=False))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
