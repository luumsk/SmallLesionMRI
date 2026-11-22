"""Deep supervision wrapper compatible with nnU-Net's ``DeepSupervisionWrapper``."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from ._utils import nnunet_ds_weights


class DeepSupervisionLoss(nn.Module):
    """``sum_i w_i * loss(outputs[i], targets[i])`` over deep-supervision outputs.

    weights : one per output; None = nnU-Net weights (1, 1/2, 1/4, ..., last 0, normalised),
        computed from the number of outputs at the first call. Zero weights are skipped.
    A plain tensor input (no deep supervision) is passed straight to ``loss``.
    """

    def __init__(self, loss: nn.Module, weights: Optional[Sequence[float]] = None) -> None:
        super().__init__()
        self.loss = loss
        self.weights = None if weights is None else tuple(float(w) for w in weights)
        if self.weights is not None and not any(w != 0 for w in self.weights):
            raise ValueError("at least one deep-supervision weight must be non-zero")

    def forward(self, outputs, targets) -> torch.Tensor:
        if not isinstance(outputs, (list, tuple)):
            return self.loss(outputs, targets)
        weights = self.weights if self.weights is not None else tuple(nnunet_ds_weights(len(outputs)))
        if len(weights) != len(outputs) or len(targets) != len(outputs):
            raise ValueError(f"{len(outputs)} outputs, {len(targets)} targets, {len(weights)} weights")
        return sum(w * self.loss(o, t) for w, o, t in zip(weights, outputs, targets) if w != 0.0)
