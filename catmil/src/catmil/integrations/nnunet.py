"""nnU-Net v2 trainers for CATMIL and the baselines of the paper.

Use with ``nnUNetv2_train DATASET CONFIG FOLD -tr <trainer>`` after ``catmil-nnunet install``
(nnU-Net only finds trainers inside its own package folder; the installer drops a stub there).

Trainers: nnUNetTrainerDiceCE, nnUNetTrainerTversky, nnUNetTrainerFocalTversky,
nnUNetTrainerCCDiceCE, nnUNetTrainerCAT, nnUNetTrainerMIL, nnUNetTrainerCATMIL.

Hyperparameters are class attributes named ``catmil_<name>``. Change them with a two-line
subclass (put it in a module inside nnunetv2/training/nnUNetTrainer/)::

    class nnUNetTrainerCATMIL_lmil05(nnUNetTrainerCATMIL):
        catmil_lambda_mil = 0.5

or with environment variables ``CATMIL_<NAME>`` (e.g. ``CATMIL_LAMBDA_MIL=0.5``) for quick
experiments. The trainers do not override ``__init__`` (nnU-Net records the constructor
arguments of the base class), so they work with every nnU-Net v2 version.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch.nn as nn

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from ..losses import (CATMILLoss, CCDiceCELoss, ComponentAdaptiveTverskyLoss, FocalTverskyLoss,
                      SmallLesionMILLoss, TverskyLoss)

__all__ = [
    "nnUNetTrainerDiceCE", "nnUNetTrainerTversky", "nnUNetTrainerFocalTversky", "nnUNetTrainerCCDiceCE",
    "nnUNetTrainerCAT", "nnUNetTrainerMIL", "nnUNetTrainerCATMIL",
]


class _Sum(nn.Module):
    def __init__(self, a: nn.Module, b: nn.Module, wa: float = 1.0, wb: float = 1.0) -> None:
        super().__init__()
        self.a, self.b, self.wa, self.wb = a, b, float(wa), float(wb)

    def forward(self, net_output, target):
        return self.wa * self.a(net_output, target) + self.wb * self.b(net_output, target)


class _CatmilTrainerMixin:
    """Shared helpers: hyperparameter lookup, deep-supervision weights, optional epoch override."""

    catmil_num_epochs = None  # None = keep nnU-Net's num_epochs

    def _hp(self, name: str) -> Any:
        default = getattr(self, f"catmil_{name}")
        env = os.environ.get(f"CATMIL_{name.upper()}")
        if env is None:
            return default
        if isinstance(default, bool):
            return env.lower() in ("1", "true", "yes")
        if default is None:
            return int(env)
        return type(default)(env)

    def _ds_weights(self) -> np.ndarray:
        w = np.array([1 / (2 ** i) for i in range(len(self._get_deep_supervision_scales()))])
        w[-1] = 0
        return w / w.sum()

    def _wrap_ds(self, loss: nn.Module) -> nn.Module:
        if self.enable_deep_supervision:
            return DeepSupervisionWrapper(loss, self._ds_weights())
        return loss

    def _activation(self) -> str:
        return "sigmoid" if self.label_manager.has_regions else "softmax"

    def initialize(self):
        if not self.was_initialized and self._hp("num_epochs") is not None:
            self.num_epochs = int(self._hp("num_epochs"))
        return super().initialize()


class nnUNetTrainerDiceCE(_CatmilTrainerMixin, nnUNetTrainer):
    """nnU-Net's default loss (DiceCE with deep supervision), under the paper's name."""


class nnUNetTrainerTversky(_CatmilTrainerMixin, nnUNetTrainer):
    """Tversky loss (alpha 0.3, beta 0.7), equivalent to MONAI TverskyLoss with softmax + one-hot."""

    catmil_alpha = 0.3
    catmil_beta = 0.7

    def _tversky_kwargs(self) -> dict:
        return dict(alpha=self._hp("alpha"), beta=self._hp("beta"),
                    ignore_index=self.label_manager.ignore_label, activation=self._activation())

    def _build_loss(self):
        return self._wrap_ds(TverskyLoss(**self._tversky_kwargs()))


class nnUNetTrainerFocalTversky(nnUNetTrainerTversky):
    """Focal Tversky loss: Tversky ** (1 / gamma), gamma 4/3."""

    catmil_focal_gamma = 4.0 / 3.0

    def _build_loss(self):
        return self._wrap_ds(FocalTverskyLoss(gamma=self._hp("focal_gamma"), **self._tversky_kwargs()))


class nnUNetTrainerCCDiceCE(_CatmilTrainerMixin, nnUNetTrainer):
    """CC-DiceCE (Bouteille et al., ISBI 2026): CC-DiceCE + global DiceCE (smooth 0), 1:1, at every
    deep-supervision output. Binary label-based training without ignore label only."""

    catmil_w_cc = 1.0
    catmil_w_global = 1.0

    def _build_loss(self):
        lm = self.label_manager
        assert not lm.has_regions and not lm.has_ignore_label and lm.num_segmentation_heads == 2, \
            "CC-DiceCE requires binary label-based training without ignore label"
        glob = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 0,
                               'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                              ignore_label=lm.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        cc = CCDiceCELoss(self.configuration_manager.spacing, self.configuration_manager.patch_size)
        return self._wrap_ds(_Sum(cc, glob, self._hp("w_cc"), self._hp("w_global")))


class nnUNetTrainerCATMIL(_CatmilTrainerMixin, nnUNetTrainer):
    """nnU-Net DiceCE + lambda_cat * CAT (linear warm-up) + lambda_mil * MIL (the paper's method).

    The base loss is nnU-Net's own (DiceCE, or Dice+BCE for regions; DDP and ignore label as in
    nnU-Net) with the usual deep supervision. CAT and MIL use the full-resolution output
    (``catmil_ds_mode = "highest"``), or every output with ``"all"``.
    """

    catmil_lambda_cat = 0.1
    catmil_lambda_mil = 0.2
    catmil_cat_warmup_epochs = 50
    catmil_alpha = 0.3
    catmil_beta = 0.7
    catmil_gamma = 1.0
    catmil_eps_cc = 5.0
    catmil_w_bg = 0.01
    catmil_mil_eps = 1e-6
    catmil_connectivity = 1
    catmil_ds_mode = "highest"
    catmil_mode = "union"          # "union" (paper), "per_class" or "auto" (per class if > 2 channels)

    def _build_loss(self):
        base = super()._build_loss()
        ds_weights = None
        if isinstance(base, DeepSupervisionWrapper):
            ds_weights, base = list(base.weight_factors), base.loss
        lm = self.label_manager
        loss = CATMILLoss(
            base_loss=base,
            lambda_cat=self._hp("lambda_cat"),
            lambda_mil=self._hp("lambda_mil"),
            cat_warmup_epochs=self._hp("cat_warmup_epochs"),
            ds_weights=ds_weights,
            ds_mode=self._hp("ds_mode"),
            cat_kwargs=dict(alpha=self._hp("alpha"), beta=self._hp("beta"), gamma=self._hp("gamma"),
                            eps_cc=self._hp("eps_cc"), w_bg=self._hp("w_bg")),
            mil_kwargs=dict(eps=self._hp("mil_eps")),
            mode=self._hp("mode"),
            activation=self._activation(),
            ignore_index=lm.ignore_label,
            connectivity=self._hp("connectivity"),
        )
        loss.set_epoch(getattr(self, "current_epoch", 0))
        self._catmil_loss = loss
        self.print_to_log_file(
            f"CATMIL loss: lambda_cat={loss.lambda_cat_final} (warm-up {loss.cat_warmup_epochs} epochs), "
            f"lambda_mil={loss.lambda_mil}, ds_mode={loss.ds_mode}, mode={self._hp('mode')}, "
            f"activation={self._activation()}, ignore_label={lm.ignore_label}")
        return loss

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        loss = getattr(self, "_catmil_loss", None)
        if loss is not None:
            loss.set_epoch(self.current_epoch)


class nnUNetTrainerCAT(nnUNetTrainerCATMIL):
    """Ablation: nnU-Net DiceCE + CAT only (per-class CAT if more than two output channels)."""

    catmil_lambda_mil = 0.0
    catmil_mode = "auto"


class nnUNetTrainerMIL(nnUNetTrainerCATMIL):
    """Ablation: nnU-Net DiceCE + MIL only."""

    catmil_lambda_cat = 0.0
    catmil_cat_warmup_epochs = 0


# re-exported so old imports from the research trainers keep working
_ = (ComponentAdaptiveTverskyLoss, SmallLesionMILLoss)
