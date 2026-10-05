# SmallLesionMRI

**Component-Adaptive and Lesion-Level Supervision for Improved Small Structure Segmentation in Brain MRI**

[![arXiv](https://img.shields.io/badge/arXiv-2604.08015-b31b1b.svg)](https://arxiv.org/abs/2604.08015)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official code for **CATMIL**, a loss function for nnU-Net that improves the detection of
small brain lesions in MRI. It adds two auxiliary terms to the standard nnU-Net loss
(Dice + cross-entropy):

- **Component-Adaptive Tversky (CAT)** reweights voxels by connected component, so small
  lesions influence the loss about as much as large ones.
- **Multiple Instance Learning (MIL)** adds lesion-level supervision that rewards detecting
  every lesion instance.

> Minh Sao Khue Luu, Evgeniy N. Pavlovskiy, Bair N. Tuchinov.
> *Component-Adaptive and Lesion-Level Supervision for Improved Small Structure
> Segmentation in Brain MRI.* arXiv:2604.08015, 2026.
> [[paper]](https://arxiv.org/abs/2604.08015)

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
  (0.7842), the best HD95 (8.10 mm) and the highest small-lesion recall (0.8554).
- **External test on 3D-MR-MS:** CATMIL again has the highest small-lesion recall
  (0.618 vs 0.537 for DiceCE), the fewest missed lesions (16.9 vs 22.0), the lowest HD95
  and the lowest FP volume. Its Dice (0.718 vs 0.734 best) and Lesion F1 (0.471 vs 0.668
  best) are the lowest of the four losses.

Per-model and per-fold outputs are in [metrics/](metrics/).

## Repository layout

```
slsseg/                    Modified nnU-Net v2 package (installs as `nnunetv2`)
  nnunetv2/training/nnUNetTrainer/
    nnUNetTrainerCATMIL.py   CATMIL (proposed)
    nnUNetTrainerCAT.py      CAT term only
    nnUNetTrainerMIL.py      MIL term only
    nnUNetTrainerTversky.py, nnUNetTrainerFocalTversky.py   loss baselines
    nnUNetTrainerSegResNet.py, ...UNETR.py, ...SwinUNETR.py,
    ...UMambaBot.py, ...UMambaEnc.py                         architecture baselines
  nnunetv2/inference/extract_features.py   feature extraction via forward hooks
scripts/                   Shell wrappers for each pipeline step and plotting scripts
compute_metrics.py         Basic per-case segmentation metrics
multiscale_eval.py         Voxel-, lesion- and size-stratified evaluation, FN/FP profiles, calibration
get_all_metrics.py         Aggregate per-fold metric files
analyze_features.py        Compare learned features between CATMIL and nnU-Net
select_test_cases_for_visualization.py   Pick improved / failure / typical cases
utils.py                   Shared helpers
metrics/                   Evaluation results reported in the paper
lesion_profile/            Per-lesion profiles for selected MSLesSeg cases
*.ipynb                    Dataset exploration, uncertainty and visualization notebooks
```

## Installation

Requires Python ≥ 3.10 and a CUDA-capable GPU for training.

```bash
git clone https://github.com/luumsk/SmallLesionMRI.git
cd SmallLesionMRI
python -m venv .venv && source .venv/bin/activate
pip install torch            # pick the build for your CUDA version: https://pytorch.org
pip install -e slsseg
```

The UMamba baselines also need `mamba-ssm` and `causal-conv1d`; CATMIL itself does not.

Set the usual nnU-Net paths (see [scripts/setvars.sh](scripts/setvars.sh)):

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

## Usage

### 1. Data

Download [MSLesSeg](https://doi.org/10.1038/s41597-025-05250-y) and convert it to the
[nnU-Net v2 dataset format](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md).
The scripts use dataset ID `333` (`Dataset333_MSLesSeg`).

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
trains for 150 epochs by default (upstream uses 1000), as in the paper. We trained on a single NVIDIA Quadro RTX 8000.

### 3. Inference

**Pretrained weights:** [download link TBA]() (CATMIL and baselines, all five folds).
Install a downloaded model with `nnUNetv2_install_pretrained_model_from_zip <file.zip>`.

```bash
nnUNetv2_predict -i $nnUNet_raw/Dataset333_MSLesSeg/imagesTs -o predictions/CATMIL/fold_0 \
  -d 333 -c 3d_fullres -f 0 -tr nnUNetTrainerCATMIL -chk checkpoint_final.pth \
  --save_probabilities
```

`--save_probabilities` is needed for the calibration and probability-based metrics below.
See [scripts/predict.sh](scripts/predict.sh).

### 4. Evaluation

```bash
python multiscale_eval.py \
  --gt_dir $nnUNet_raw/Dataset333_MSLesSeg/labelsTs \
  --pred_mask_dir predictions/CATMIL/fold_0 \
  --pred_prob_dir predictions/CATMIL/fold_0 \
  --out_csv metrics/multiscale_eval_CATMIL_fold0.csv \
  --out_json metrics/multiscale_eval_CATMIL_fold0.json \
  --small_voxels_thresh 150
```

This reports Dice, HD95 and ASSD, lesion-wise detection (6-connected components,
any-voxel overlap), recall by lesion size (tiny ≤10, small 11–50, medium 51–200,
large >200 voxels), false-negative and false-positive profiles, and calibration
(entropy, Brier score, ECE). [scripts/multiscale_eval.sh](scripts/multiscale_eval.sh)
loops over all models and folds, and [scripts/get_all_metrics.sh](scripts/get_all_metrics.sh)
aggregates the results.

### 5. Figures and analysis

| Script | Output |
| --- | --- |
| [scripts/plot_recall_by_size.py](scripts/plot_recall_by_size.py) | Lesion recall per size bin |
| [scripts/plot_fn_per_case_paired.py](scripts/plot_fn_per_case_paired.py) | Missed lesions per case, CATMIL vs. each baseline |
| [scripts/build_lesion_tables.py](scripts/build_lesion_tables.py) | Per-lesion and per-component tables across thresholds |
| [scripts/plot_lesion_figures.py](scripts/plot_lesion_figures.py) | Recall–size curves, lesion fate, operating curves, lesion gallery |
| [scripts/extract_features.sh](scripts/extract_features.sh), [scripts/analyze_features.sh](scripts/analyze_features.sh) | Feature-space comparison of CATMIL and nnU-Net |

The shell scripts contain absolute paths from our setup; edit them before running.

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
