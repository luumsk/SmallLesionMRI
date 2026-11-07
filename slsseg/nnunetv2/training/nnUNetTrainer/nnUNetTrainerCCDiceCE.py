"""nnU-Net trainer with the CC-DiceCE loss (Bouteille et al., ISBI 2026, arXiv:2511.17146).

Port of ``nnUNetTrainerGlobalCCDiceCE`` from the official code
(https://github.com/TIO-IKIM/Learning-to-Look-Closer, commit 62b317b,
``training/loss/instance_losses.py`` and ``nnUNetTrainerInstanceLosses.py``), i.e. the
"CC-DiceCE" objective of the paper:

    loss = DeepSupervision( 1.0 * CC-DiceCE + 1.0 * global DiceCE(smooth=0) )

CC-DiceCE per sample: label the ground-truth foreground into 26-connected components,
assign every voxel to its nearest component (Voronoi region, EDT with the voxel spacing
of the current deep-supervision level), and for each region compute
``1 - 2*sum(p*y)/sum(p+y)`` + mean voxel CE over the region; average over components.
A sample without foreground gets the fallback ``1 + mean CE over the whole patch``.
The batch loss is the mean over samples. As in the official code, the CC term is applied
at every deep-supervision output (on the downsampled targets) and the global DiceCE uses
smooth=0 instead of nnU-Net's default 1e-5.

Plans, splits, optimiser and the 150 epochs come from nnUNetTrainer, so the run is
comparable with nnUNetTrainer and nnUNetTrainerCATMIL on Dataset333_MSLesSeg.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

try:  # GPU path, as in the official code
    import cupy as cp
    from cupyx.scipy.ndimage import distance_transform_edt as _cp_edt
    from cupyx.scipy.ndimage import label as _cp_label
    _HAS_CUPY = True
except Exception:  # CPU fallback (scipy), same result
    _HAS_CUPY = False

_EPS = 1e-7

# torch.compiler exists from torch 2.1; older versions do not compile the loss anyway
_no_compile = getattr(getattr(torch, "compiler", None), "disable", lambda f: f)


def _voronoi_cupy(fg: torch.Tensor, sampling) -> torch.Tensor:
    # uint8 (not bool) for DLPack: older torch versions cannot export bool tensors
    m = cp.from_dlpack(fg.to(torch.uint8).contiguous()) > 0
    lab, _ = _cp_label(m, structure=cp.ones((3, 3, 3), dtype=bool))
    if int(lab.max()) == 0:
        return torch.zeros(fg.shape, dtype=torch.int64, device=fg.device)
    idx = _cp_edt(lab == 0, sampling=sampling, return_distances=False, return_indices=True,
                  float64_distances=False)
    return torch.from_dlpack(lab[tuple(idx)].astype(cp.int64)).to(fg.device)


def _voronoi(fg: torch.Tensor, sampling) -> torch.Tensor:
    """Voronoi map (0 = no component in patch, else 1..K) of a binary 3D mask; int64 on fg.device."""
    global _HAS_CUPY
    if _HAS_CUPY and fg.is_cuda:
        try:
            return _voronoi_cupy(fg, sampling)
        except Exception as e:  # e.g. cupy/CUDA mismatch: use scipy (same result) for the rest of the run
            print(f"CC-DiceCE: cupy Voronoi failed ({e!r}); falling back to scipy on CPU", flush=True)
            _HAS_CUPY = False
    m = fg.cpu().numpy()
    lab, n = ndimage.label(m, structure=np.ones((3, 3, 3), dtype=bool))
    if n == 0:
        return torch.zeros(fg.shape, dtype=torch.int64, device=fg.device)
    idx = ndimage.distance_transform_edt(lab == 0, sampling=sampling, return_distances=False,
                                         return_indices=True)
    return torch.from_numpy(lab[tuple(idx)].astype(np.int64)).to(fg.device)


class CCDiceCELoss(nn.Module):
    """Voronoi-based component-wise DiceCE (official ``VectorizedCCLoss``, mode='dice_ce')."""

    def __init__(self, spacing, patch_size) -> None:
        super().__init__()
        self.spacing = tuple(float(s) for s in spacing)
        self.patch_size = tuple(int(s) for s in patch_size)

    def _sampling(self, shape):
        # spacing of the current deep-supervision level
        return tuple(s * full / cur for s, full, cur in zip(self.spacing, self.patch_size, shape))

    @_no_compile
    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert net_output.ndim == 5 and net_output.shape[1] == 2, "binary 3D only"
        y = target[:, 0].long()                       # (B, X, Y, Z)
        probs = torch.softmax(net_output.float(), 1)
        p_fg = probs[:, 1]
        y_fg = (y == 1).float()
        ce_map = -torch.log(probs.clamp_min(_EPS)).gather(1, y.unsqueeze(1)).squeeze(1)
        sampling = self._sampling(y.shape[1:])

        losses = []
        for b in range(y.shape[0]):
            vor = _voronoi(y[b] == 1, sampling)
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


class _Sum(nn.Module):
    def __init__(self, cc: nn.Module, glob: nn.Module, w_cc: float = 1.0, w_glob: float = 1.0) -> None:
        super().__init__()
        self.cc, self.glob, self.w_cc, self.w_glob = cc, glob, w_cc, w_glob

    def forward(self, net_output, target):
        return self.w_cc * self.cc(net_output, target) + self.w_glob * self.glob(net_output, target)


class nnUNetTrainerCCDiceCE(nnUNetTrainer):
    """CC-DiceCE + global DiceCE (1:1), both under deep supervision (official GlobalCCDiceCE)."""

    def _build_loss(self) -> nn.Module:
        lm = self.label_manager
        assert not lm.has_regions and not lm.has_ignore_label and lm.num_segmentation_heads == 2, \
            "CC-DiceCE requires binary label-based training without ignore label"
        glob = DC_and_CE_loss({'batch_dice': self.configuration_manager.batch_dice, 'smooth': 0,
                               'do_bg': False, 'ddp': self.is_ddp}, {}, weight_ce=1, weight_dice=1,
                              ignore_label=lm.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)
        cc = CCDiceCELoss(self.configuration_manager.spacing, self.configuration_manager.patch_size)
        loss = _Sum(cc, glob)
        if self.enable_deep_supervision:
            weights = np.array([1 / (2 ** i) for i in range(len(self._get_deep_supervision_scales()))])
            weights[-1] = 0
            loss = DeepSupervisionWrapper(loss, weights / weights.sum())
        return loss
