"""Baseline losses compared in the paper: DiceCE, Tversky, Focal Tversky and CC-DiceCE.

* ``DiceCELoss`` reproduces nnU-Net v2's default ``DC_and_CE_loss`` (memory-efficient soft Dice
  without background + cross entropy). Like nnU-Net, the Dice term is ``-Dice``, so the loss is
  ``CE - Dice`` (a constant shift from ``CE + 1 - Dice``; gradients are identical).
* ``TverskyLoss`` reproduces ``monai.losses.TverskyLoss(to_onehot_y=True, softmax=True)``.
* ``FocalTverskyLoss`` is ``Tversky ** (1 / gamma)`` as used in the paper (gamma = 4/3).
* ``CCDiceCELoss`` is the Voronoi component-wise DiceCE of Bouteille et al. (ISBI 2026,
  arXiv:2511.17146), ported from the official code; ``CCDiceCEGlobalLoss`` adds the global
  DiceCE (smooth = 0) 1:1, which is the "CC-DiceCE" objective of that paper.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

from ._utils import resolve_activation, squeeze_target

_EPS = 1e-7


def _one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """(B, *spatial) long -> (B, C, *spatial) float."""
    return F.one_hot(labels.clamp_min(0), n_classes).movedim(-1, 1).to(torch.float32)


def _region_target(logits: torch.Tensor, target: torch.Tensor, ignore_index):
    """Region targets (B, R, *sp) and loss mask (B, 1, *sp) or None (nnU-Net: ignore = last channel)."""
    if logits.shape[1] == 1 and (target.ndim == logits.ndim - 1 or target.shape[1] == 1):
        tgt = squeeze_target(target)
        mask = None if ignore_index is None else (tgt != int(ignore_index)).unsqueeze(1).float()
        return (tgt > 0).unsqueeze(1).float() * (1.0 if mask is None else mask), mask
    if ignore_index is not None:
        return target[:, :-1].float(), (1 - target[:, -1:]).float()
    return target.float(), None


class SoftDiceLoss(nn.Module):
    """nnU-Net ``MemoryEfficientSoftDiceLoss`` on probabilities; returns ``-mean Dice``."""

    def __init__(self, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.0) -> None:
        super().__init__()
        self.batch_dice, self.do_bg, self.smooth = bool(batch_dice), bool(do_bg), float(smooth)

    def forward(self, x: torch.Tensor, y_onehot: torch.Tensor,
                loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        axes = tuple(range(2, x.ndim))
        if not self.do_bg:
            x, y_onehot = x[:, 1:], y_onehot[:, 1:]
        with torch.no_grad():
            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)
        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)
        if self.batch_dice:
            intersect, sum_pred, sum_gt = intersect.sum(0), sum_pred.sum(0), sum_gt.sum(0)
        dc = (2 * intersect + self.smooth) / torch.clip(sum_gt + sum_pred + self.smooth, 1e-8)
        return -dc.mean()


class DiceCELoss(nn.Module):
    """nnU-Net v2 default loss (soft Dice + CE for softmax; soft Dice + BCE for sigmoid/regions).

    Defaults match nnU-Net: smooth 1e-5, background excluded from Dice for softmax, included for
    sigmoid. ``batch_dice`` should follow the nnU-Net plans (True for 3d_fullres usually).
    """

    def __init__(self, batch_dice: bool = False, smooth: float = 1e-5, do_bg: Optional[bool] = None,
                 weight_ce: float = 1.0, weight_dice: float = 1.0, ignore_index: Optional[int] = None,
                 activation: str = "auto") -> None:
        super().__init__()
        self.batch_dice, self.smooth, self.do_bg = batch_dice, smooth, do_bg
        self.weight_ce, self.weight_dice = float(weight_ce), float(weight_dice)
        self.ignore_index, self.activation = ignore_index, activation

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        act = resolve_activation(net_output, self.activation)
        if act == "sigmoid":
            dice = SoftDiceLoss(self.batch_dice, True if self.do_bg is None else self.do_bg, self.smooth)
            tgt, mask = _region_target(net_output, target, self.ignore_index)
            dc = dice(torch.sigmoid(net_output), tgt, loss_mask=mask)
            if mask is None:
                ce = F.binary_cross_entropy_with_logits(net_output, tgt)
            else:
                ce = (F.binary_cross_entropy_with_logits(net_output, tgt, reduction="none") * mask).sum() \
                    / torch.clip(mask.sum(), 1e-8)
            return self.weight_ce * ce + self.weight_dice * dc

        dice = SoftDiceLoss(self.batch_dice, False if self.do_bg is None else self.do_bg, self.smooth)
        tgt = squeeze_target(target)
        n_cls = int(net_output.shape[1])
        mask, tgt_dice = None, tgt
        if self.ignore_index is not None:
            mask = (tgt != int(self.ignore_index)).unsqueeze(1)
            tgt_dice = torch.where(mask[:, 0], tgt, torch.zeros_like(tgt))
        dc = dice(torch.softmax(net_output, 1), _one_hot(tgt_dice, n_cls), loss_mask=mask)
        if self.ignore_index is not None and not bool(mask.any()):
            ce = net_output.new_zeros(())
        else:
            ce = F.cross_entropy(net_output, tgt,
                                 ignore_index=-100 if self.ignore_index is None else int(self.ignore_index))
        return self.weight_ce * ce + self.weight_dice * dc


class TverskyLoss(nn.Module):
    """Soft Tversky loss, ``1 - TP / (TP + alpha FP + beta FN)`` per sample and class, averaged.

    Equivalent to ``monai.losses.TverskyLoss(to_onehot_y=True, softmax=True)`` (background
    included, smooth 1e-5) for softmax outputs; per channel on sigmoid outputs.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, include_background: bool = True,
                 smooth_nr: float = 1e-5, smooth_dr: float = 1e-5, ignore_index: Optional[int] = None,
                 activation: str = "auto") -> None:
        super().__init__()
        self.alpha, self.beta = float(alpha), float(beta)
        self.include_background = include_background
        self.smooth_nr, self.smooth_dr = float(smooth_nr), float(smooth_dr)
        self.ignore_index, self.activation = ignore_index, activation

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        act = resolve_activation(net_output, self.activation)
        if act == "sigmoid":
            p = torch.sigmoid(net_output)
            g, mask = _region_target(net_output, target, self.ignore_index)
        else:
            tgt = squeeze_target(target)
            mask = None
            if self.ignore_index is not None:
                mask = (tgt != int(self.ignore_index)).unsqueeze(1).float()
                tgt = torch.where(mask[:, 0] > 0, tgt, torch.zeros_like(tgt))
            p = torch.softmax(net_output, 1)
            g = _one_hot(tgt, int(net_output.shape[1]))
            if not self.include_background:
                p, g = p[:, 1:], g[:, 1:]
        if mask is not None:
            p, g = p * mask, g * mask
        dims = tuple(range(2, p.ndim))
        tp = (p * g).sum(dims)
        fp = self.alpha * (p * (1 - g)).sum(dims)
        fn = self.beta * ((1 - p) * g).sum(dims)
        return (1.0 - (tp + self.smooth_nr) / (tp + fp + fn + self.smooth_dr)).mean()


class FocalTverskyLoss(nn.Module):
    """``TverskyLoss ** (1 / gamma)`` (paper: alpha 0.3, beta 0.7, gamma 4/3)."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, gamma: float = 4.0 / 3.0, **tversky_kwargs) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, **tversky_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.pow(self.tversky(net_output, target), 1.0 / self.gamma)


# ----------------------------------------------------------------------------- CC-DiceCE

try:  # GPU Voronoi as in the official code
    import cupy as _cp
    from cupyx.scipy.ndimage import distance_transform_edt as _cp_edt
    from cupyx.scipy.ndimage import label as _cp_label
    _HAS_CUPY = True
except Exception:  # scipy on CPU, same result
    _HAS_CUPY = False

_no_compile = getattr(getattr(torch, "compiler", None), "disable", lambda f: f)


def _voronoi_cupy(fg: torch.Tensor, sampling) -> torch.Tensor:
    m = _cp.from_dlpack(fg.to(torch.uint8).contiguous()) > 0  # uint8: old torch cannot export bool
    lab, _ = _cp_label(m, structure=_cp.ones((3,) * m.ndim, dtype=bool))
    if int(lab.max()) == 0:
        return torch.zeros(fg.shape, dtype=torch.int64, device=fg.device)
    idx = _cp_edt(lab == 0, sampling=sampling, return_distances=False, return_indices=True,
                  float64_distances=False)
    return torch.from_dlpack(lab[tuple(idx)].astype(_cp.int64)).to(fg.device)


def voronoi_components(fg: torch.Tensor, sampling=None) -> torch.Tensor:
    """Voronoi map of a binary mask: every voxel gets the id of its nearest GT component.

    Fully connected components (26 in 3D), Euclidean distance with ``sampling`` (voxel spacing).
    0 = no component in the patch. Returns int64 on ``fg.device``.
    """
    global _HAS_CUPY
    if _HAS_CUPY and fg.is_cuda:
        try:
            return _voronoi_cupy(fg, sampling)
        except Exception as e:  # e.g. cupy/CUDA mismatch: scipy for the rest of the run
            print(f"CC-DiceCE: cupy Voronoi failed ({e!r}); falling back to scipy on CPU", flush=True)
            _HAS_CUPY = False
    m = fg.detach().cpu().numpy()
    lab, n = ndimage.label(m, structure=np.ones((3,) * m.ndim, dtype=bool))
    if n == 0:
        return torch.zeros(fg.shape, dtype=torch.int64, device=fg.device)
    idx = ndimage.distance_transform_edt(lab == 0, sampling=sampling, return_distances=False,
                                         return_indices=True)
    return torch.from_numpy(lab[tuple(idx)].astype(np.int64)).to(fg.device)


class CCDiceCELoss(nn.Module):
    """Voronoi component-wise DiceCE (official ``VectorizedCCLoss``, mode ``dice_ce``).

    For each GT component, its Voronoi region gets ``1 - 2 sum(p y) / sum(p + y)`` plus the mean
    voxel CE in the region; the sample loss is the mean over components. A sample without
    foreground gets ``1 + mean CE``. Binary softmax (2 channels) only, as in the official code.

    spacing : voxel spacing at full resolution (default 1 per axis).
    patch_size : full-resolution patch size; used to scale the spacing at deep-supervision levels.
        If None, every input is treated as full resolution.
    """

    def __init__(self, spacing: Optional[Sequence[float]] = None,
                 patch_size: Optional[Sequence[int]] = None) -> None:
        super().__init__()
        self.spacing = None if spacing is None else tuple(float(s) for s in spacing)
        self.patch_size = None if patch_size is None else tuple(int(s) for s in patch_size)

    def _sampling(self, shape):
        spacing = self.spacing or (1.0,) * len(shape)
        if self.patch_size is None:
            return spacing
        return tuple(s * full / cur for s, full, cur in zip(spacing, self.patch_size, shape))

    @_no_compile
    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if net_output.shape[1] != 2:
            raise ValueError("CCDiceCELoss supports binary softmax outputs (2 channels) only")
        y = squeeze_target(target)
        probs = torch.softmax(net_output.float(), 1)
        p_fg = probs[:, 1]
        y_fg = (y == 1).float()
        ce_map = -torch.log(probs.clamp_min(_EPS)).gather(1, y.unsqueeze(1)).squeeze(1)
        sampling = self._sampling(y.shape[1:])

        losses = []
        for b in range(y.shape[0]):
            vor = voronoi_components(y[b] == 1, sampling)
            k = int(vor.max())
            if k == 0:  # official fallback: Dice term 1, CE over the whole patch
                losses.append(ce_map[b].mean() + 1.0)
                continue
            ids = vor.reshape(-1)

            def seg_sum(v):
                return torch.zeros(k + 1, device=v.device, dtype=v.dtype).scatter_add_(0, ids, v.reshape(-1))[1:]

            inter = seg_sum(p_fg[b] * y_fg[b])
            denom = seg_sum(p_fg[b]) + seg_sum(y_fg[b])
            count = seg_sum(torch.ones_like(p_fg[b]))
            comp = (1.0 - 2.0 * inter / denom.clamp_min(_EPS)) + seg_sum(ce_map[b]) / count.clamp_min(1.0)
            valid = (count > 0).float()
            losses.append((comp * valid).sum() / valid.sum().clamp_min(1.0))
        return torch.stack(losses).mean()


class CCDiceCEGlobalLoss(nn.Module):
    """The paper's CC-DiceCE objective: ``w_cc * CCDiceCE + w_global * DiceCE(smooth=0)``.

    In nnU-Net it is applied at every deep-supervision output (wrap it in ``DeepSupervisionLoss``).
    """

    def __init__(self, spacing=None, patch_size=None, batch_dice: bool = False,
                 w_cc: float = 1.0, w_global: float = 1.0, global_loss: Optional[nn.Module] = None) -> None:
        super().__init__()
        self.cc = CCDiceCELoss(spacing, patch_size)
        self.glob = global_loss if global_loss is not None else DiceCELoss(batch_dice=batch_dice, smooth=0.0)
        self.w_cc, self.w_global = float(w_cc), float(w_global)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.w_cc * self.cc(net_output, target) + self.w_global * self.glob(net_output, target)
