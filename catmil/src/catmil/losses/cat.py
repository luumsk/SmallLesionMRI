"""Component-Adaptive Tversky (CAT) loss."""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..cc import label_batch
from ._utils import BinaryProblem, binary_problems, no_autocast, no_compile, squeeze_target


def component_weight_map(
    components: torch.Tensor,
    fg: torch.Tensor,
    valid: Optional[torch.Tensor],
    gamma: float,
    eps_cc: float,
    w_bg: float,
) -> torch.Tensor:
    """Per-voxel weights (B, *spatial): (|c| + eps_cc)^-gamma inside GT component c, w_bg outside.

    Ignored voxels (``valid == False``) get weight 0.
    """
    w = torch.full(fg.shape, float(w_bg), dtype=torch.float32, device=fg.device)
    for b in range(fg.shape[0]):
        lab = components[b].long()
        sizes = torch.bincount(lab.reshape(-1)).to(torch.float32)
        inv = (sizes + float(eps_cc)).pow(-float(gamma))
        inv[0] = 0.0
        w[b] = torch.where(fg[b], inv[lab], w[b])
    if valid is not None:
        w = w * valid.to(w.dtype)
    return w


class ComponentAdaptiveTverskyLoss(nn.Module):
    """Tversky loss in which each ground-truth lesion counts about equally.

    Foreground voxels of a lesion with |c| voxels get weight ``(|c| + eps_cc) ** -gamma``,
    background voxels get ``w_bg``. With gamma = 1 every lesion contributes roughly the same
    total weight, so large lesions no longer dominate the gradient.

    Parameters
    ----------
    alpha, beta : weights of false positives and false negatives (beta > alpha favours recall).
    gamma : size exponent; larger values emphasise small lesions more.
    eps_cc : added to the component size, stabilises one-voxel lesions.
    w_bg : weight of background voxels.
    smooth : numerical smoothing in the Tversky ratio.
    connectivity : component connectivity (1 = 6-neighbourhood in 3D, the paper's setting).
    mode : ``"union"`` (paper), ``"per_class"`` or ``"auto"``; see ``binary_problems``.
    class_ids : foreground classes for ``"per_class"`` (default all).
    activation : ``"auto"`` (sigmoid for 1 channel, else softmax), ``"softmax"`` or ``"sigmoid"``.
    ignore_index : ignored label (weight 0, never part of a lesion).

    Defaults are the values used in the paper (alpha=0.3, beta=0.7, gamma=1, eps_cc=5, w_bg=0.01).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        eps_cc: float = 5.0,
        w_bg: float = 0.01,
        smooth: float = 1e-5,
        connectivity: int = 1,
        mode: str = "union",
        class_ids: Optional[Sequence[int]] = None,
        activation: str = "auto",
        ignore_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.alpha, self.beta, self.gamma = float(alpha), float(beta), float(gamma)
        self.eps_cc, self.w_bg, self.smooth = float(eps_cc), float(w_bg), float(smooth)
        self.connectivity = int(connectivity)
        self.mode, self.class_ids, self.activation = mode, class_ids, activation
        self.ignore_index = ignore_index

    def _tversky(self, p: BinaryProblem, components: Optional[torch.Tensor]) -> torch.Tensor:
        lab = components if components is not None else label_batch(p.fg, self.connectivity)
        w = component_weight_map(lab, p.fg, p.valid, self.gamma, self.eps_cc, self.w_bg).unsqueeze(1)
        gt = p.fg.unsqueeze(1).to(torch.float32)
        dims = list(range(2, p.prob.ndim))
        tp = (w * p.prob * gt).sum(dim=dims)
        fp = (w * p.prob * (1.0 - gt)).sum(dim=dims)
        fn = (w * (1.0 - p.prob) * gt).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()

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
            losses = [self._tversky(p, components) for p in problems if bool(p.fg.any())]
            if not losses:  # no lesion in the patch
                return net_output.new_zeros((), dtype=torch.float32)
            return torch.stack(losses).mean()
