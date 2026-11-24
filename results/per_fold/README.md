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

Every file here except `MSLesSeg/lambda_sweep/multiscale_eval_CAT01MIL02_foldall_all.csv`
was written by `bash scripts/multiscale_eval.sh` (which writes to
`results/per_fold_rerun/` by default), and `analysis/make_tables.py` builds the
paper tables from them. The CAT01MIL02 predictions are not in the data
mirror, so that file comes from the original evaluation.

The brier, ECE and entropy columns are not reported in any document. In the
CAT01MIL02 file they are wrong (probabilities read in the wrong axis order,
fixed in `multiscale_eval.py` since). The `Ljubljana/all/*_pp5` files now hold
metrics on masks that were actually filtered; they equal the 3D-MR-MS PP5
values from `analysis/pp5_3dmrms_twolevel.py` (lesion F1: CATMIL 0.6302,
DiceCE 0.6867).

`analysis/stats_significance.py results/per_fold results/lesion_tables/lesions.csv results/stats`
reproduces `results/stats/` from these files.
