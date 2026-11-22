"""Lesion-level multiple-instance-learning (MIL) loss."""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ..cc import label_batch
from ._utils import binary_problems, no_autocast, no_compile, squeeze_target


def component_max(prob: torch.Tensor, components: torch.Tensor) -> torch.Tensor:
    """Differentiable max of ``prob`` (*spatial) inside each component of ``components`` (*spatial).

    Returns a 1D tensor with one value per non-empty component (label order). One pass over the
    volume (``scatter_reduce``), instead of one full-volume mask per component.
    """
    lab = components.reshape(-1).long()
    k = int(lab.max()) if lab.numel() else 0
    if k == 0:
        return prob.new_zeros(0)
    vals = prob.reshape(-1)
    out = vals.new_zeros(k + 1).scatter_reduce(0, lab, vals, reduce="amax", include_self=False)
    present = torch.bincount(lab, minlength=k + 1) > 0
    present[0] = False
    return out[present]


class SmallLesionMILLoss(nn.Module):
    """Each ground-truth lesion is a bag; it is "found" if one voxel in it has high probability.

    loss = mean over all lesions c in the batch of ``-log(max_{v in c} p(v))``.

    The term is independent of lesion size, so a missed one-voxel lesion costs as much as a missed
    large one. It is meant to be added to a voxel loss with a small weight (paper: 0.2).

    Parameters
    ----------
    eps : lower clamp of the max probability before the log.
    connectivity, mode, class_ids, activation, ignore_index : as in ``ComponentAdaptiveTverskyLoss``.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        connectivity: int = 1,
        mode: str = "union",
        class_ids: Optional[Sequence[int]] = None,
        activation: str = "auto",
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.connectivity = int(connectivity)
        self.mode, self.class_ids, self.activation = mode, class_ids, activation
        self.ignore_index = ignore_index

    @no_compile
    def forward(self, net_output: torch.Tensor, target: torch.Tensor,
                components: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``components``: optional precomputed GT component labels (B, *spatial), union mode only."""
        with no_autocast(net_output.device.type):
            problems = binary_problems(net_output, target, activation=self.activation, mode=self.mode,
                                       class_ids=self.class_ids, ignore_index=self.ignore_index)
            if components is not None and len(problems) != 1:
                raise ValueError("precomputed components are only supported with one binary problem")
            if components is not None:
                components = squeeze_target(components).to(net_output.device)
            terms: List[torch.Tensor] = []
            for p in problems:
                if not bool(p.fg.any()):
                    continue
                lab = components if components is not None else label_batch(p.fg, self.connectivity)
                if components is not None:  # only GT foreground voxels belong to a lesion
                    lab = torch.where(p.fg, lab.long(), torch.zeros_like(lab, dtype=torch.long))
                for b in range(p.prob.shape[0]):
                    m = component_max(p.prob[b, 0], lab[b])
                    if m.numel():
                        terms.append(-torch.log(m.clamp_min(self.eps)))
            if not terms:
                return net_output.new_zeros((), dtype=torch.float32)
            return torch.cat(terms).mean()
