"""Patient-level bootstrap of small-lesion recall on both test sets.

Reads metrics/lesion_tables/lesions.csv (written by build_lesion_tables.py) and,
for each loss, computes small-lesion recall (lesions <= 150 voxels) as in
multiscale_eval.py: recall per scan, averaged over scans, then over the five
fold models. This reproduces the reported values (e.g. CATMIL 0.8730 on
MSLesSeg, 0.618 on 3D-MR-MS).

The test patients are resampled with replacement (all scans of a patient are
kept together) and the recall of each loss and the paired difference
CATMIL - baseline are recomputed for each resample; 95% percentile intervals
are printed.

Usage: python recall_bootstrap.py [n_resamples]
"""
import collections
import csv
import os
import random
import statistics as st
import sys

LESIONS = os.path.join(os.path.dirname(__file__), os.pardir, "metrics", "lesion_tables", "lesions.csv")
LOSSES = ["CATMIL", "DiceCE", "Tversky", "FocalTversky"]
SMALL_VOX = 150


def main(n_resamples=2000):
    rows = list(csv.DictReader(open(LESIONS)))
    for ds in ["MSLesSeg", "3D-MR-MS"]:
        # (loss, fold, case) -> [detected small lesions, small lesions]
        acc = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            if r["dataset"] != ds or int(r["size_vox"]) > SMALL_VOX:
                continue
            k = (r["model"], r["fold"], r["case"])
            acc[k][0] += r["detected"] == "True"
            acc[k][1] += 1
        folds = sorted(set(k[1] for k in acc))
        cases = sorted(set(k[2] for k in acc))
        # MSLesSeg cases are timepoints (P13_T1, P13_T2, ...) of one patient
        patient = (lambda c: c.split("_")[0]) if ds == "MSLesSeg" else (lambda c: c)
        patients = sorted(set(patient(c) for c in cases))

        def recall(loss, cs):
            return st.mean(st.mean(acc[(loss, f, c)][0] / acc[(loss, f, c)][1] for c in cs) for f in folds)

        random.seed(0)
        samples = []
        for _ in range(n_resamples):
            cs = [c for p in random.choices(patients, k=len(patients)) for c in cases if patient(c) == p]
            samples.append({m: recall(m, cs) for m in LOSSES})
        lo, hi = int(0.025 * n_resamples), int(0.975 * n_resamples) - 1

        print(ds, "patients", len(patients), "scans", len(cases))
        for m in LOSSES:
            v = sorted(s[m] for s in samples)
            print(f"  {m:13s} {recall(m, cases):.4f}  95% CI [{v[lo]:.3f}, {v[hi]:.3f}]")
        for m in LOSSES[1:]:
            d = sorted(s["CATMIL"] - s[m] for s in samples)
            diff = recall("CATMIL", cases) - recall(m, cases)
            print(f"  CATMIL - {m:13s} {diff:+.4f}  95% CI [{d[lo]:+.3f}, {d[hi]:+.3f}]")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)
