"""Input handling shared by all losses.

All losses take raw network logits ``(B, C, *spatial)`` and a target in nnU-Net layout:

* label-based training: integer labels ``(B, 1, *spatial)`` or ``(B, *spatial)`` (float or long);
* region-based training (sigmoid, nnU-Net ``has_regions``): binary ``(B, R, *spatial)``, and with an
  ignore label one extra last channel that marks ignored voxels (nnU-Net convention).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch


def squeeze_target(target: torch.Tensor) -> torch.Tensor:
    """(B, 1, *spatial) or (B, *spatial) labels -> (B, *spatial) long."""
    if target.ndim >= 2 and target.shape[1] == 1:
        target = target[:, 0]
    return target.long()


def foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    """Foreground probability (B, 1, *spatial): sigmoid for C=1, else sum of softmax[1:]."""
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    return torch.softmax(logits, dim=1)[:, 1:].sum(dim=1, keepdim=True)


def resolve_activation(logits: torch.Tensor, activation: str) -> str:
    if activation == "auto":
        return "sigmoid" if logits.shape[1] == 1 else "softmax"
    if activation not in ("softmax", "sigmoid"):
        raise ValueError(f"activation must be 'auto', 'softmax' or 'sigmoid', got {activation!r}")
    return activation


@dataclass
class BinaryProblem:
    """One foreground-vs-rest problem extracted from a (multi-class) prediction."""
    prob: torch.Tensor                 # (B, 1, *spatial) float32, differentiable
    fg: torch.Tensor                   # (B, *spatial) bool, ground-truth foreground
    valid: Optional[torch.Tensor]      # (B, *spatial) bool, False = ignored voxel; None = all valid
    class_id: Optional[int] = None     # None = union of all foreground classes


def binary_problems(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    activation: str = "auto",
    mode: str = "union",
    class_ids: Optional[Sequence[int]] = None,
    ignore_index: Optional[int] = None,
) -> List[BinaryProblem]:
    """Split a prediction into binary lesion problems for the component-based losses.

    mode
        ``"union"``: one problem, foreground = any non-background class (the paper's setting).
        ``"per_class"``: one problem per foreground class (``class_ids``, default 1..C-1).
        ``"auto"``: ``"per_class"`` if C > 2 softmax channels, else ``"union"``.
        With sigmoid outputs and C > 1 (nnU-Net regions) every region is its own problem.
    ignore_index
        Label of ignored voxels (label-based), or, for regions, any non-None value means the last
        target channel is the ignore mask (nnU-Net convention).
    """
    logits = logits.float()
    act = resolve_activation(logits, activation)
    n_ch = int(logits.shape[1])

    if act == "sigmoid" and n_ch > 1:  # region-based: binary target per channel
        tgt = target
        valid = None
        if ignore_index is not None:
            valid = tgt[:, -1] == 0
            tgt = tgt[:, :-1]
        if tgt.shape[1] != n_ch:
            raise ValueError(f"region target has {tgt.shape[1]} channels, logits have {n_ch}")
        probs = torch.sigmoid(logits)
        ids = range(n_ch) if class_ids is None else class_ids
        return [BinaryProblem(probs[:, c:c + 1], (tgt[:, c] > 0) & (valid if valid is not None else True),
                              valid, int(c)) for c in ids]

    tgt = squeeze_target(target)
    valid = None if ignore_index is None else tgt != int(ignore_index)

    def _fg(mask: torch.Tensor) -> torch.Tensor:
        return mask if valid is None else mask & valid

    if act == "sigmoid":  # single channel
        return [BinaryProblem(torch.sigmoid(logits), _fg(tgt > 0), valid, None)]

    probs = torch.softmax(logits, dim=1)
    if mode == "auto":
        mode = "per_class" if n_ch > 2 else "union"
    if mode == "union":
        return [BinaryProblem(probs[:, 1:].sum(dim=1, keepdim=True), _fg(tgt > 0), valid, None)]
    if mode != "per_class":
        raise ValueError(f"mode must be 'union', 'per_class' or 'auto', got {mode!r}")
    ids = range(1, n_ch) if class_ids is None else [int(c) for c in class_ids if 0 <= int(c) < n_ch]
    return [BinaryProblem(probs[:, c:c + 1], _fg(tgt == c), valid, int(c)) for c in ids]


def no_autocast(device_type: str):
    """Context manager that disables autocast so component sums are done in float32."""
    if device_type not in ("cuda", "cpu", "mps", "xpu"):
        device_type = "cpu"
    try:
        return torch.autocast(device_type=device_type, enabled=False)
    except Exception:  # pragma: no cover - old torch / unsupported device
        return torch.autocast(device_type="cpu", enabled=False)


def nnunet_ds_weights(n_outputs: int) -> np.ndarray:
    """nnU-Net deep-supervision weights: 1, 1/2, 1/4, ..., lowest output 0, normalised to sum 1."""
    w = np.array([1 / (2 ** i) for i in range(n_outputs)], dtype=np.float64)
    if n_outputs > 1:
        w[-1] = 0
    return w / w.sum()


# The component losses call scipy on the CPU; keep them out of torch.compile graphs
# (nnU-Net >= 2.5 compiles the loss). torch.compiler exists from torch 2.1.
no_compile = getattr(getattr(torch, "compiler", None), "disable", lambda f: f)
