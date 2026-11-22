"""CATMIL: base voxel loss + lambda_cat * CAT + lambda_mil * MIL."""
from __future__ import annotations

import warnings
from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from ._utils import nnunet_ds_weights
from .baselines import DiceCELoss
from .cat import ComponentAdaptiveTverskyLoss
from .mil import SmallLesionMILLoss


class CATMILLoss(nn.Module):
    """The paper's objective: ``L = L_base + lambda_cat(t) * L_CAT + lambda_mil * L_MIL``.

    ``lambda_cat`` ramps linearly from 0 to its final value over ``cat_warmup_epochs`` epochs;
    call ``loss.set_epoch(epoch)`` at the start of every epoch (0-based). With
    ``cat_warmup_epochs=0`` the full weight is used from the start.

    Parameters
    ----------
    base_loss : voxel loss applied to a single output; default nnU-Net DiceCE.
    lambda_cat, lambda_mil : final weights of the CAT and MIL terms (paper: 0.1 and 0.2).
    cat_warmup_epochs : epochs of linear warm-up for lambda_cat (paper: 50).
    ds_weights : deep-supervision weights for list inputs; None = nnU-Net weights.
    ds_mode : with list inputs, ``"highest"`` applies CAT and MIL to the full-resolution output
        only (paper), ``"all"`` applies them at every output with ``ds_weights``.
    cat_kwargs, mil_kwargs : extra arguments for ``ComponentAdaptiveTverskyLoss`` /
        ``SmallLesionMILLoss`` (alpha, beta, gamma, eps_cc, w_bg, ...).
    mode, activation, ignore_index, connectivity : passed to both CAT and MIL.

    ``forward(outputs, targets, components=None, return_components=False)`` takes a tensor or a
    deep-supervision list. With ``return_components=True`` it returns a dict with ``loss`` (the
    differentiable total) and detached ``base``, ``cat``, ``mil``, ``lambda_cat`` for logging.
    """

    def __init__(
        self,
        base_loss: Optional[nn.Module] = None,
        lambda_cat: float = 0.1,
        lambda_mil: float = 0.2,
        cat_warmup_epochs: int = 50,
        ds_weights: Optional[Sequence[float]] = None,
        ds_mode: str = "highest",
        cat_kwargs: Optional[dict] = None,
        mil_kwargs: Optional[dict] = None,
        mode: str = "union",
        activation: str = "auto",
        ignore_index: Optional[int] = None,
        connectivity: int = 1,
    ) -> None:
        super().__init__()
        if ds_mode not in ("highest", "all"):
            raise ValueError(f"ds_mode must be 'highest' or 'all', got {ds_mode!r}")
        common = dict(mode=mode, activation=activation, ignore_index=ignore_index, connectivity=connectivity)
        self.base_loss = base_loss if base_loss is not None else DiceCELoss(ignore_index=ignore_index,
                                                                            activation=activation)
        self.cat_loss = ComponentAdaptiveTverskyLoss(**{**common, **(cat_kwargs or {})})
        self.mil_loss = SmallLesionMILLoss(**{**common, **(mil_kwargs or {})})
        self.lambda_cat_final = float(lambda_cat)
        self.lambda_mil = float(lambda_mil)
        self.cat_warmup_epochs = int(cat_warmup_epochs)
        self.ds_weights = None if ds_weights is None else tuple(float(w) for w in ds_weights)
        self.ds_mode = ds_mode
        self._epoch_set = self.cat_warmup_epochs <= 0
        self.lambda_cat = self.lambda_cat_final if self._epoch_set else 0.0

    def set_epoch(self, epoch: int) -> float:
        """Update lambda_cat for a 0-based epoch; returns the new value."""
        self._epoch_set = True
        if self.cat_warmup_epochs <= 0:
            self.lambda_cat = self.lambda_cat_final
        else:
            t = min(max(int(epoch), 0), self.cat_warmup_epochs) / float(self.cat_warmup_epochs)
            self.lambda_cat = float(t * self.lambda_cat_final)
        return self.lambda_cat

    def _aux(self, out: torch.Tensor, tgt: torch.Tensor, components):
        zero = out.new_zeros((), dtype=torch.float32)
        # lambda 0 (warm-up start): skip the term entirely, also avoids 0 * NaN
        cat = self.cat_loss(out, tgt, components) if self.lambda_cat > 0 else zero
        mil = self.mil_loss(out, tgt, components) if self.lambda_mil > 0 else zero
        return cat, mil

    def forward(self, outputs, targets, components: Optional[torch.Tensor] = None,
                return_components: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if not self._epoch_set:
            warnings.warn("CATMILLoss: set_epoch() was never called, so lambda_cat stays 0 "
                          "(CAT disabled). Call loss.set_epoch(epoch) every epoch, or pass "
                          "cat_warmup_epochs=0.", stacklevel=2)
            self._epoch_set = True

        if isinstance(outputs, (list, tuple)):
            weights = self.ds_weights if self.ds_weights is not None else tuple(nnunet_ds_weights(len(outputs)))
            base = sum(w * self.base_loss(o, t) for w, o, t in zip(weights, outputs, targets) if w != 0.0)
            if self.ds_mode == "highest":
                cat, mil = self._aux(outputs[0], targets[0], components)
            else:
                if components is not None:
                    raise ValueError("precomputed components are only supported with ds_mode='highest'")
                cat = mil = 0.0
                for w, o, t in zip(weights, outputs, targets):
                    if w != 0.0:
                        c, m = self._aux(o, t, None)
                        cat, mil = cat + w * c, mil + w * m
        else:
            base = self.base_loss(outputs, targets)
            cat, mil = self._aux(outputs, targets, components)

        total = base + self.lambda_cat * cat + self.lambda_mil * mil
        if not return_components:
            return total
        as_t = lambda v: v.detach() if torch.is_tensor(v) else torch.tensor(float(v))  # noqa: E731
        return {"loss": total, "base": as_t(base), "cat": as_t(cat), "mil": as_t(mil),
                "lambda_cat": torch.tensor(self.lambda_cat)}
