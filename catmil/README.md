# catmil

Loss functions for **small lesion segmentation** in 3D (and 2D) medical images:

* **CATMIL**: the method of *Component-Adaptive and Lesion-Level Supervision for Improved Small
  Structure Segmentation in Brain MRI* (Luu, Pavlovskiy, Tuchinov). It is DiceCE plus a
  component-adaptive Tversky term (CAT), so every lesion counts about equally, plus a lesion-level
  MIL term that penalises missed lesions regardless of their size.
* **The baselines from the paper**, for fair comparison: DiceCE (nnU-Net default), Tversky, Focal
  Tversky, and CC-DiceCE (Bouteille et al., ISBI 2026).
* **nnU-Net v2 trainers** for all of them, and lesion-wise metrics and tools.

Every loss takes raw logits `(B, C, *spatial)` and nnU-Net-style targets `(B, 1, *spatial)`. The
losses work in plain PyTorch, MONAI or nnU-Net. Tests check that they reproduce the paper's
implementation (values and gradients).

## Install

```bash
pip install -e ./catmil                 # torch, numpy, scipy
pip install -e "./catmil[nnunet]"       # + nnU-Net v2 trainers
pip install -e "./catmil[fast-cc]"      # + faster connected components (cc3d)
```

## PyTorch / MONAI

```python
from catmil import CATMILLoss

loss_fn = CATMILLoss()          # DiceCE + 0.1*CAT (linear warm-up over 50 epochs) + 0.2*MIL
for epoch in range(n_epochs):
    loss_fn.set_epoch(epoch)    # drives the CAT warm-up
    for image, label in loader:              # label: (B, 1, D, H, W) integer map
        logits = model(image)                # (B, C, D, H, W); or a list for deep supervision
        out = loss_fn(logits, label, return_components=True)
        out["loss"].backward()
        log(base=out["base"], cat=out["cat"], mil=out["mil"])
```

Pick any loss by name, for example to compare losses from a config file:

```python
from catmil import get_loss, list_losses
list_losses()   # ['cat', 'catmil', 'cc_dice_ce', 'cc_dice_ce_component', 'dice_ce', 'focal_tversky', 'mil', 'tversky']
loss_fn = get_loss("focal_tversky", deep_supervision=True)   # takes lists of outputs/targets
```

Other options:

| option | meaning |
|---|---|
| `CATMILLoss(base_loss=...)` | use your own voxel loss instead of DiceCE |
| `ds_mode="highest"` / `"all"` | CAT and MIL on the full-resolution output only (paper), or on every deep-supervision output |
| `mode="union"` / `"per_class"` / `"auto"` | one lesion mask = any foreground (paper), or one per class (multi-class, e.g. BraTS) |
| `activation="sigmoid"` | region-based / multi-label outputs (binary target per channel) |
| `ignore_index=k` | voxels labelled `k` are ignored (never part of a lesion) |
| `cat_kwargs=dict(alpha, beta, gamma, eps_cc, w_bg)` | CAT hyperparameters |
| `components=` | precomputed GT components from `LabelComponentsd` in the data loader (avoids a GPU→CPU copy per step) |

Paper defaults: alpha 0.3, beta 0.7, gamma 1, eps_cc 5, w_bg 0.01, lambda_cat 0.1 (50-epoch
warm-up), lambda_mil 0.2, 6-connectivity.

## nnU-Net v2

```bash
pip install -e "./catmil[nnunet]"
catmil-nnunet install        # puts a stub in nnunetv2/training/nnUNetTrainer/ (nnU-Net only looks there)
catmil-nnunet status         # shows which module each trainer resolves to
nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNetTrainerCATMIL
```

Trainers: `nnUNetTrainerCATMIL`, `nnUNetTrainerCAT`, `nnUNetTrainerMIL` (ablations),
`nnUNetTrainerDiceCE`, `nnUNetTrainerTversky`, `nnUNetTrainerFocalTversky` and
`nnUNetTrainerCCDiceCE`.

* The base term is nnU-Net's own loss (DiceCE, or Dice+BCE for regions), with nnU-Net's deep
  supervision, DDP batch Dice and ignore label.
* The CAT warm-up follows `current_epoch`, including when you resume from a checkpoint.
* CAT and MIL run in float32 even under AMP, and stay out of `torch.compile`.
* Tested with nnU-Net 2.1.1 (the paper's fork in `slsseg/`) and 2.8.1 (PyPI).

Change hyperparameters with a subclass. Put it in a file inside
`nnunetv2/training/nnUNetTrainer/`:

```python
from catmil.integrations.nnunet import nnUNetTrainerCATMIL

class nnUNetTrainerCATMIL_lmil05(nnUNetTrainerCATMIL):
    catmil_lambda_mil = 0.5          # any catmil_* attribute, see catmil/integrations/nnunet.py
```

For a quick test you can use environment variables instead, e.g.
`CATMIL_LAMBDA_MIL=0.5 nnUNetv2_train ...`. `CATMIL_NUM_EPOCHS` overrides the epoch count.
Without it, nnU-Net's own default applies (1000 in stock nnU-Net, 150 in the paper's fork).

`catmil-nnunet uninstall` removes the stub. Run `install` again after you upgrade nnunetv2.

## Lesion-wise evaluation

Voxel Dice hides missed small lesions. These functions use the paper's definitions:
6-connected components, and a lesion counts as detected if any predicted voxel overlaps it.

```python
from catmil.metrics import lesion_detection_metrics, recall_by_size, lesion_table
lesion_detection_metrics(gt, pred, spacing=(1, 1, 1))   # lesion F1/precision/recall, small-lesion recall, FP count, missed volume
recall_by_size(gt, pred)                                 # tiny <=10, small <=50, medium <=200, large voxels
lesion_table(gt, pred, prob)                             # one row per lesion: size, detected, coverage, max prob
```

## Command line

```bash
catmil profile path/to/labelsTr     # lesion-size distribution: does your dataset have many small lesions?
catmil bench --shape 2 2 128 128 128 --device cuda   # ms per forward+backward for each loss
catmil losses
```

## Tests

```bash
pip install -e "./catmil[test]"
pytest catmil/tests      # the nnU-Net tests run when nnunetv2 is installed
```

`tests/_reference_losses.py` holds frozen copies of the research code. The parity tests compare
values and gradients against it, against nnU-Net's `DC_and_CE_loss` / `DC_and_BCE_loss`, against
MONAI's `TverskyLoss`, and against the paper's evaluation script.

## Citation

See `CITATION.cff` in the repository root. If you use CC-DiceCE, please also cite Bouteille et al.,
ISBI 2026 (arXiv:2511.17146).
