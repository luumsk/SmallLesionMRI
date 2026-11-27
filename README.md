# Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI

[![arXiv](https://img.shields.io/badge/arXiv-2604.08015-b31b1b.svg)](https://arxiv.org/abs/2604.08015)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository holds the code, results and LaTeX sources needed to reproduce the paper.
**CATMIL** is a loss function for nnU-Net that improves the detection of small brain lesions
in MRI. It adds two auxiliary terms to the standard nnU-Net loss (Dice + cross-entropy):

- **Component-Adaptive Tversky (CAT)** reweights voxels by connected component, so small
  lesions influence the loss about as much as large ones.
- **Multiple Instance Learning (MIL)** adds lesion-level supervision that rewards detecting
  every lesion instance.

> Minh Sao Khue Luu, Evgeniy N. Pavlovskiy, Bair N. Tuchinov.
> *Component-Adaptive and Lesion-Level Supervision for Improved Small Structure
> Segmentation in Brain MRI.* arXiv:2604.08015, 2026.
> [[paper]](https://arxiv.org/abs/2604.08015)

**To use CATMIL in your own project**, install the loss package instead:
[comploss](https://github.com/luumsk/comploss) (`pip install comploss`). It contains CATMIL, CAT,
MIL, CC-DiceCE and the baseline losses, nnU-Net v2 trainers for each, and object-level metrics.
This repository pins the exact version used for the paper.

## Results

MSLesSeg held-out test set (6 patients). All losses use the same nnU-Net `3d_fullres`
setup, and each value is the mean ± std over the five fold models. Small lesions have
≤150 voxels (`--small_voxels_thresh 150` in `multiscale_eval.py`, 1 mm isotropic).

| Metric | **CATMIL (ours)** | DiceCE (nnU-Net) | Tversky | FocalTversky |
| --- | --- | --- | --- | --- |
| Small Lesion Recall ↑ | **0.8730 ± 0.01** | 0.7956 ± 0.03 | 0.8336 ± 0.02 | 0.8313 ± 0.02 |
| FN Count ↓ | **2.57 ± 0.15** | 4.95 ± 0.40 | 3.77 ± 0.23 | 3.70 ± 0.24 |
| FN Volume Fraction ↓ | **0.0214 ± 0.003** | 0.0341 ± 0.01 | 0.0257 ± 0.01 | 0.0250 ± 0.005 |
| Dice ↑ | **0.7834 ± 0.01** | 0.7796 ± 0.01 | 0.7706 ± 0.01 | 0.7802 ± 0.01 |
| HD95 (mm) ↓ | **7.98 ± 1.84** | 9.04 ± 1.34 | 10.24 ± 0.15 | 8.21 ± 1.93 |
| Lesion F1 ↑ | 0.7571 ± 0.02 | 0.8433 ± 0.01 | 0.8402 ± 0.01 | **0.8455 ± 0.01** |
| FP Volume (mm³) ↓ | **1537 ± 245** | 1819 ± 654 | 2621 ± 559 | 2282 ± 710 |

- CATMIL misses about 48% fewer lesions than DiceCE. Recall on tiny lesions (≤10 voxels)
  rises from 0.14 to 0.33.
- **Trade-off:** CATMIL predicts many small disconnected components. Most of them lie
  within 1–2 mm of a real lesion. They add little false-positive volume but lower
  Lesion F1. If components smaller than 5 voxels are removed, CATMIL's Lesion F1 rises
  to 0.8415, compared with 0.8586 for DiceCE. CATMIL then still has the best Dice
  (0.7842), the best HD95 (8.10 mm) and the highest small-lesion recall (0.8553).
- **External test on 3D-MR-MS:** CATMIL again has the highest small-lesion recall
  (0.618 vs 0.537 for DiceCE), the fewest missed lesions (16.9 vs 22.0), the lowest HD95
  and the lowest FP volume. Its Dice (0.718 vs 0.734 best) and Lesion F1 (0.471 vs 0.668
  best) are the lowest of the four losses.

Per-model and per-fold outputs are in [results/](results/).

## Repository layout

```
paper/
  arxiv/              LaTeX source of the arXiv version (main_v02.tex = arXiv v5), figures/, compile.sh
  journal/            LaTeX source of the journal version (MDPI), compile.sh
slsseg/               Modified nnU-Net v2 used in the paper (installs as `nnunetv2`, 150 epochs by default)
  nnunetv2/training/nnUNetTrainer/
    nnUNetTrainerCATMIL.py, ...CAT.py, ...MIL.py,           loss trainers: thin stubs that import
    ...Tversky.py, ...FocalTversky.py, ...CCDiceCE.py       the trainers from comploss
    ...SegResNet.py, ...UNETR.py, ...SwinUNETR.py,
    ...UMambaBot.py, ...UMambaEnc.py                        architecture baselines
  nnunetv2/inference/extract_features.py                    feature extraction via forward hooks
evaluation/           Metric code: multiscale_eval.py (voxel-, lesion- and size-stratified metrics,
                      FN/FP profiles, calibration), compute_metrics.py, get_all_metrics.py, utils.py
analysis/             One script per analysis, table or figure (see "Reproducing the analyses")
scripts/              Shell pipeline: config.sh (paths), preprocess, train, predict, evaluate
results/              Results reported in the paper (CSV); results/analyses_2026-11/ holds the
                      architecture comparison, out-of-fold and PP5 analyses added in November 2026
notebooks/            Dataset exploration, uncertainty and visualization notebooks
requirements.txt      Pinned environment (slsseg, comploss, analysis packages)
```

## Installation

Requires Python ≥ 3.10 and a CUDA-capable GPU for training.

```bash
git clone https://github.com/luumsk/SmallLesionMRI.git
cd SmallLesionMRI
python -m venv .venv && source .venv/bin/activate
pip install torch                  # pick the build for your CUDA version: https://pytorch.org
pip install -r requirements.txt    # slsseg (nnunetv2 2.1.1), comploss 0.1.1, analysis packages
```

PyTorch: the paper's models were trained with torch 2.0.1 (CUDA 11.8), as recorded in each fold's
`debug.json`. To match that setup, install `torch==2.0.1` (e.g. `pip install torch==2.0.1
--index-url https://download.pytorch.org/whl/cu118`). Newer versions work too: inference and
evaluation were re-run with torch 2.15 and reproduce the stored predictions, and comploss is tested
with torch 2.0.1–2.15. Retraining matches the paper up to normal GPU nondeterminism, not to the last
digit.

The UMamba baselines also need `mamba-ssm` and `causal-conv1d`; CATMIL itself does not.
A few analysis scripts need extra packages, listed at the end of `requirements.txt`.

### Data location

All data stay outside the repository, in one folder that we call `SLS_DATA_ROOT`:

```
$SLS_DATA_ROOT/
  nnUNet_raw/  nnUNet_preprocessed/  nnUNet_results/          the usual nnU-Net folders
  SmallLesionMRI/MSLesSeg/final_checkpoints/<MODEL>/fold_<k>/  test-set predictions (MSLesSeg)
  SmallLesionMRI/LjubljanaMS/final_checkpoints/...             test-set predictions (3D-MR-MS)
```

Set it once before running anything:

```bash
export SLS_DATA_ROOT=/path/to/data
```

[scripts/config.sh](scripts/config.sh) derives `nnUNet_raw`, `nnUNet_preprocessed` and
`nnUNet_results` from it (each can be overridden), and every script in `scripts/` sources it. The
Python scripts in `analysis/` read the same variable.

## Reproducing the paper

### 1. Data

Download [MSLesSeg](https://doi.org/10.1038/s41597-025-05250-y) and convert it to the
[nnU-Net v2 dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).
The scripts use dataset ID `333` (`Dataset333_MSLesSeg`) and, for the external test set
3D-MR-MS, `666` (`Dataset666_LjubljanaMS`).

```bash
nnUNetv2_plan_and_preprocess -d 333 --verify_dataset_integrity
```

### 2. Training

Select a loss or architecture with `-tr`:

```bash
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train 333 3d_fullres $FOLD -tr nnUNetTrainerCATMIL
done
```

[scripts/train.sh](scripts/train.sh) trains every model in the paper. The bundled nnU-Net
trains for 150 epochs by default (upstream uses 1000), as in the paper. We trained on a single
NVIDIA Quadro RTX 8000.

### 3. Inference

**Pretrained weights:** [doi:10.5281/zenodo.24591106](https://doi.org/10.5281/zenodo.24591106)
(all MSLesSeg models of the paper: 7 losses and 4 architectures × 5 folds, and the 4 loss-weight
models; CC BY 4.0). The 3D-MR-MS models are available on request, since that dataset's licence
(CC BY-NC-ND) may not allow sharing derived models publicly.
Install a downloaded model with `nnUNetv2_install_pretrained_model_from_zip <file.zip>`.

```bash
nnUNetv2_predict -i $nnUNet_raw/Dataset333_MSLesSeg/imagesTs \
  -o $SLS_DATA_ROOT/SmallLesionMRI/MSLesSeg/final_checkpoints/CATMIL/fold_0 \
  -d 333 -c 3d_fullres -f 0 -tr nnUNetTrainerCATMIL -chk checkpoint_final.pth \
  --save_probabilities
```

`--save_probabilities` is needed for the calibration and probability-based metrics below.
[scripts/predict.sh](scripts/predict.sh) loops over models and folds.

### 4. Evaluation

```bash
python evaluation/multiscale_eval.py \
  --gt_dir $nnUNet_raw/Dataset333_MSLesSeg/labelsTs \
  --pred_mask_dir $SLS_DATA_ROOT/SmallLesionMRI/MSLesSeg/final_checkpoints/CATMIL/fold_0 \
  --pred_prob_dir $SLS_DATA_ROOT/SmallLesionMRI/MSLesSeg/final_checkpoints/CATMIL/fold_0 \
  --out_csv results/MSLesSeg/multiscale_eval_CATMIL_fold0.csv \
  --out_json results/MSLesSeg/multiscale_eval_CATMIL_fold0.json \
  --small_voxels_thresh 150 --metrics_mode all --allow_missing_prob
```

This reports Dice, HD95 and ASSD, lesion-wise detection (6-connected components,
any-voxel overlap), recall by lesion size (tiny ≤10, small 11–50, medium 51–200,
large >200 voxels), false-negative and false-positive profiles, and calibration
(entropy, Brier score, ECE). `--metrics_mode main` (the default) writes only the main paper
metrics; `--out_json` is optional.

To regenerate everything in [results/per_fold/](results/per_fold/) (all losses, PP3/PP5/PP10/PP15,
the lambda sweep and 3D-MR-MS), and then the paper tables:

```bash
bash scripts/multiscale_eval.sh                        # -> results/per_fold_rerun/ (OUT_ROOT=...)
python analysis/make_tables.py results/per_fold_rerun  # -> results/tables/
```

The script first rebuilds the post-processed masks with
[evaluation/postprocess_min_size.py](evaluation/postprocess_min_size.py) (predicted 6-connected
components below N voxels removed). [analysis/make_tables.py](analysis/make_tables.py) writes the
mean ± std, paired-difference and ablation tables of the paper as LaTeX `tabular`s with the
unrounded values next to them. The paired differences come from `results/stats/`; rerun
[stats_significance.py](analysis/stats_significance.py) on the new folder first if you want them
recomputed too.

### 5. Reproducing the analyses

Run the scripts from the repository root (`python analysis/<script>.py ...`). Scripts marked
*data* read predictions or labels from `SLS_DATA_ROOT`; the others need only files in `results/`.

| Script | What it computes | Input | Output |
| --- | --- | --- | --- |
| [build_lesion_tables.py](analysis/build_lesion_tables.py) | Per-lesion and per-component tables over probability thresholds, both test sets | data | `results/lesion_tables/` |
| [recall_bootstrap.py](analysis/recall_bootstrap.py) | Patient-level bootstrap of lesion-level metrics | lesion tables | printed |
| [stats_significance.py](analysis/stats_significance.py) | Significance of CATMIL vs each baseline (case and lesion level) | per-fold metric CSVs, lesion tables | `results/stats/` |
| [error_analysis.py](analysis/error_analysis.py) | Which lesions are missed, by which loss, how close the misses are | lesion tables | `results/error_analysis/` |
| [operating_curves.py](analysis/operating_curves.py) | Detection/precision operating curves | lesion tables | `results/operating_curves/` |
| [plot_lesion_figures.py](analysis/plot_lesion_figures.py) | Recall–size curves, lesion fate, operating curves, lesion gallery | lesion tables, data | paper figures |
| [plot_fn_per_case_paired.py](analysis/plot_fn_per_case_paired.py) | Missed lesions per case, CATMIL vs each baseline | lesion tables | paper figure |
| [plot_recall_by_size.py](analysis/plot_recall_by_size.py) | Lesion recall per size bin | per-fold metric JSONs | figure |
| [lesion_extent.py](analysis/lesion_extent.py) | Largest linear extent of every lesion (MAGNIMS size definition) | data | CSV |
| [periventricular_flip.py](analysis/periventricular_flip.py), [periventricular_summary.py](analysis/periventricular_summary.py) | Do missed small lesions change the periventricular indicator? | data | `results/periventricular/` |
| [pv_independence_check.py](analysis/pv_independence_check.py) | Does losing the periventricular indicator follow the independence model? | `results/periventricular/` | `results/analyses_2026-11/` |
| [ablation_fn_count.py](analysis/ablation_fn_count.py) | Missed lesions per scan for the CAT / MIL ablation | data | `results/analyses_2026-11/` |
| [architecture_vs_loss.py](analysis/architecture_vs_loss.py), [architecture_voxel_metrics.py](analysis/architecture_voxel_metrics.py) | Architecture (SegResNet, SwinUNETR, UNETR, UMambaBot) vs loss | data | `results/analyses_2026-11/` |
| [architecture_vs_loss_twolevel.py](analysis/architecture_vs_loss_twolevel.py) | The same comparison with the paper's two-level bootstrap | `results/analyses_2026-11/` | `results/analyses_2026-11/` |
| [pp5_3dmrms_twolevel.py](analysis/pp5_3dmrms_twolevel.py), [pp5_case_metrics.py](analysis/pp5_case_metrics.py) | Results after removing predicted components < 5 voxels (PP5) | lesion tables / data | `results/analyses_2026-11/` |
| [make_tables.py](analysis/make_tables.py) | Paper tables from the per-fold metrics and `results/stats/` | `results/per_fold/`, `results/stats/` | `results/tables/` |
| [combine_lesion_profile.py](analysis/combine_lesion_profile.py) | Combined ground-truth lesion profile (volume, contrast, sharpness, KMeans cluster) | `results/lesion_profile/MSLesSeg/` | `results/lesion_profile/MSLesSeg_lesion_profile.csv` |
| [oof_evaluation.py](analysis/oof_evaluation.py) | Out-of-fold evaluation on the 47 cross-validation patients | data | `results/analyses_2026-11/` |
| [extract_features.sh](scripts/extract_features.sh), [analyze_features.py](analysis/analyze_features.py) | Feature-space comparison of CATMIL and nnU-Net | data | — |
| [select_test_cases_for_visualization.py](analysis/select_test_cases_for_visualization.py) | Pick improved, failure and typical cases for the qualitative figures | per-fold metric CSVs | CSV |

### 6. Building the paper

```bash
bash paper/arxiv/compile.sh          # -> paper/arxiv/submission/v02/ (PDF + arXiv tarball)
bash paper/journal/compile.sh        # needs the MDPI template in paper/MDPI_template_ACS/
```

The figures the paper includes are in [paper/arxiv/figures/](paper/arxiv/figures/).

## Code versions

The tag [`arxiv-v5`](https://github.com/luumsk/SmallLesionMRI/tree/arxiv-v5) holds the code and
LaTeX source exactly as of the arXiv v5 submission (before this repository was reorganised and
the losses moved to comploss). The losses in comploss reproduce that implementation; its parity
tests compare values and gradients against a frozen copy of the original code.

## Citation

If you use this code, please cite:

```bibtex
@article{luu2026catmil,
  title   = {Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI},
  author  = {Luu, Minh Sao Khue and Pavlovskiy, Evgeniy N. and Tuchinov, Bair N.},
  journal = {arXiv preprint arXiv:2604.08015},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.08015}
}
```

GitHub's "Cite this repository" button uses [CITATION.cff](CITATION.cff).

## Acknowledgements

Built on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). The architecture baselines use
[MONAI](https://github.com/Project-MONAI/MONAI) and [U-Mamba](https://github.com/bowang-lab/U-Mamba).

## License

Released under the [MIT License](LICENSE). The nnU-Net code in `slsseg/` is a modified copy
of nnU-Net and remains under its original
[Apache License 2.0](https://github.com/MIC-DKFZ/nnUNet/blob/master/LICENSE).

**For research use only. Not approved for clinical use.**
