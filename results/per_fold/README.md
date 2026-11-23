# Per-fold, per-case metrics

All tables in the paper and the dissertation are computed from these files.
Each CSV has one row per test scan for one fold model and is written by
`evaluation/multiscale_eval.py` (lesions are 6-connected components, small
lesions have at most 150 voxels). Table values are the mean and sample
standard deviation over the five fold models of the per-fold case means.
The JSON next to each CSV holds those case means rounded to five decimals;
compute tables from the CSVs, because rounding the JSON values again can
change the last printed digit.

| Folder | Test set | Contents |
|---|---|---|
| `MSLesSeg/main/` | MSLesSeg, 12 scans | Main metrics for each loss, `_pp5` after removing predicted components under 5 voxels, `CATMILpp{3,10,15}` for other size thresholds |
| `MSLesSeg/all/` | MSLesSeg, 12 scans | Same evaluation with `--metrics_mode all` (adds the false-positive analysis columns) |
| `MSLesSeg/lambda_sweep/` | MSLesSeg, 12 scans | Ablation on the CAT and MIL weights; `CAT0aMIL0b` is lambda_CAT = 0.a, lambda_MIL = 0.b, one model trained on all training data (`fold_all`) |
| `Ljubljana/all/` | 3D-MR-MS, 6 scans | All metrics for each loss, with and without PP5 |

Model names: `nnUNet` is DiceCE, `CCDiceCE` is CC-DiceCE.

The brier, ECE and entropy columns are not reported in any document and
should not be used: they were computed with the probability array in the
wrong axis order.

`analysis/stats_significance.py results/per_fold results/lesion_tables/lesions.csv results/stats`
reproduces `results/stats/` from these files.
