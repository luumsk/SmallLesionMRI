"""Is the small-lesion problem solved by a different architecture, or by the loss?

The repository already holds 5-fold predictions on the 12 MSLesSeg test scans
for four other architectures trained with the unchanged nnU-Net default loss
(DiceCE) and the same 150-epoch budget: SegResNet, SwinUNETR, UNETR and
UMambaBot (trainers in slsseg/nnunetv2/training/nnUNetTrainer; none overrides
the loss). UMambaEnc has only an ensemble prediction and is left out. No model
is trained here.

Step 1 builds per-lesion / per-component tables for these architectures with
exactly the code of scripts/build_lesion_tables.py (6-connected components,
any-voxel overlap, masks = p > 0.5).
Step 2 computes, as scripts/recall_bootstrap.py does, small-lesion recall
(<= 150 voxels), missed lesions per scan and lesion F1, averaged over scans and
then over the five fold models, together with the existing DiceCE nnU-Net and
CATMIL nnU-Net rows of metrics/lesion_tables/lesions.csv, and a patient-level
bootstrap (2000 resamples of the 6 test patients) of the paired differences
  (a) CATMIL nnU-Net - each architecture with DiceCE   (change of loss)
  (b) each architecture with DiceCE - DiceCE nnU-Net   (change of architecture)

Usage: python architecture_vs_loss.py [n_workers] [--skip-tables] [--pp5]
  --pp5: remove predicted components < 5 voxels first (PP5, as in recall_bootstrap.py --pp5)
Outputs: analyses_2026-11/results/arch_lesion_tables/{lesions,components}.csv,
         analyses_2026-11/results/architecture_vs_loss_{summary,bootstrap}.csv
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import build_lesion_tables as blt  # noqa: E402

OUT = os.path.join(REPO, "analyses_2026-11", "results")
TABLES_NEW = os.path.join(OUT, "arch_lesion_tables")
TABLES_OLD = os.path.join(REPO, "metrics", "lesion_tables")
ARCHS = {"SegResNet": "SegResNet", "SwinUNETR": "SwinUNETR", "UNETR": "UNETR", "UMambaBot": "UMambaBot"}
SMALL_VOX, EPS = 150, 1e-8
KEYS = ["dataset", "model", "fold", "case"]
METRICS = ["small_recall", "fn_count", "lesion_f1"]


def _job(job):
    blt.MODELS = ARCHS
    blt.DATASETS = {"MSLesSeg": blt.DATASETS["MSLesSeg"]}
    return blt.process(job)


def build_tables(workers):
    os.makedirs(TABLES_NEW, exist_ok=True)
    pred_dir, gt_dir = blt.DATASETS["MSLesSeg"]
    cases = sorted(f[:-7] for f in os.listdir(f"{blt.ROOT}/{gt_dir}") if f.endswith(".nii.gz"))
    # SwinUNETR has no fold_4 prediction in final_checkpoints; folds without predictions are skipped
    # (SwinUNETR is therefore averaged over 4 fold models).
    jobs = [("MSLesSeg", m, f, c) for m in ARCHS for f in blt.FOLDS for c in cases
            if os.path.exists(f"{blt.ROOT}/{pred_dir}/{ARCHS[m]}/fold_{f}/{c}.npz")]
    skipped = sorted({(m, f) for m in ARCHS for f in blt.FOLDS
                      if not os.path.exists(f"{blt.ROOT}/{pred_dir}/{ARCHS[m]}/fold_{f}")
                      or not os.listdir(f"{blt.ROOT}/{pred_dir}/{ARCHS[m]}/fold_{f}")})
    print("skipped (model, fold):", skipped, flush=True)
    with Pool(workers) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    pd.concat([r[0] for r in res]).to_csv(f"{TABLES_NEW}/lesions.csv", index=False)
    pd.concat([r[1] for r in res]).to_csv(f"{TABLES_NEW}/components.csv", index=False)


def per_scan(tables, pp_min_vox=0):
    les = pd.read_csv(os.path.join(tables, "lesions.csv"))
    comp = pd.read_csv(os.path.join(tables, "components.csv"))
    les, comp = les[les.dataset == "MSLesSeg"], comp[(comp.dataset == "MSLesSeg") & (comp.threshold == 0.5)]
    if pp_min_vox:  # PP5 exactly as scripts/recall_bootstrap.py
        les = les.assign(detected=les["hit_size_t0.5"] >= pp_min_vox)
        comp = comp[comp.size_vox >= pp_min_vox]
    les = les.assign(small=les.size_vox <= SMALL_VOX)
    les["small_hit"] = les.small & les.detected
    g = les.groupby(KEYS).agg(gt_n=("detected", "size"), tp_gt=("detected", "sum"),
                              small_n=("small", "sum"), small_hit=("small_hit", "sum"))
    p = comp.groupby(KEYS).agg(pr_n=("matched", "size"), tp_pr=("matched", "sum"))
    df = g.join(p).fillna(0)
    rec, prec = df.tp_gt / (df.gt_n + EPS), df.tp_pr / (df.pr_n + EPS)
    df["lesion_f1"] = 2 * prec * rec / (prec + rec + EPS)
    df["fn_count"] = df.gt_n - df.tp_gt
    df["small_recall"] = df.small_hit / (df.small_n + EPS)
    return df[METRICS].reset_index()


def main(workers=4, skip_tables=False, n_boot=2000, pp5=False):
    if not skip_tables:
        build_tables(workers)
    pp = 5 if pp5 else 0
    sfx = "_pp5" if pp5 else ""
    old = per_scan(TABLES_OLD, pp)
    old = old[old.model.isin(["CATMIL", "DiceCE"])].replace({"model": {"DiceCE": "nnU-Net (DiceCE)",
                                                                       "CATMIL": "nnU-Net (CATMIL)"}})
    new = per_scan(TABLES_NEW, pp)
    new["model"] = new.model + " (DiceCE)"
    df = pd.concat([old, new])
    df["patient"] = df.case.str.split("_").str[0]
    per_case = df.groupby(["model", "patient", "case"])[METRICS].mean()
    models = ["nnU-Net (CATMIL)", "nnU-Net (DiceCE)"] + [f"{a} (DiceCE)" for a in ARCHS]
    patients = sorted(df.patient.unique())
    cases_of = {p: sorted(df[df.patient == p].case.unique()) for p in patients}
    all_cases = [c for p in patients for c in cases_of[p]]

    def value(m, cases):
        return per_case.loc[m].reset_index(level=0, drop=True).loc[cases].mean()

    summary = pd.DataFrame({m: value(m, all_cases) for m in models}).T
    summary.index.name = "model"
    summary.to_csv(os.path.join(OUT, f"architecture_vs_loss_summary{sfx}.csv"))
    print(summary.round(4).to_string())

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        cases = [c for p in rng.choice(patients, len(patients)) for c in cases_of[p]]
        boot.append({m: value(m, cases) for m in models})
    rows = []
    for metric in METRICS:
        for m in models[2:]:
            for kind, a, b in (("loss change: CATMIL nnU-Net - arch DiceCE", "nnU-Net (CATMIL)", m),
                               ("architecture change: arch DiceCE - nnU-Net DiceCE", m, "nnU-Net (DiceCE)")):
                diff = np.array([x[a][metric] - x[b][metric] for x in boot])
                rows.append({"metric": metric, "comparison": kind, "a": a, "b": b,
                             "diff": summary.loc[a, metric] - summary.loc[b, metric],
                             "ci_lo": np.quantile(diff, 0.025), "ci_hi": np.quantile(diff, 0.975)})
    bt = pd.DataFrame(rows)
    bt.to_csv(os.path.join(OUT, f"architecture_vs_loss_bootstrap{sfx}.csv"), index=False)
    print(bt.round(4).to_string(index=False))


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(int(a[0]) if a else 4, "--skip-tables" in sys.argv, pp5="--pp5" in sys.argv)
