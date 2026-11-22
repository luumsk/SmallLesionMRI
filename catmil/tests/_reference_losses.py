"""Verbatim copies of the research losses, for parity tests (do not edit).

CAT / MIL / Tversky / FocalTversky: main @ f780984, slsseg/nnunetv2/training/nnUNetTrainer/.
CC-DiceCE: branch ccdicece-baseline, nnUNetTrainerCCDiceCE.py. nnU-Net imports and trainer classes removed.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

# ---- nnUNetTrainerCAT.py
def _safe_squeeze_target(target: torch.Tensor) -> torch.Tensor:
    """Normalize nnUNet targets to integer label maps.

    nnUNet targets are typically shaped (B, 1, *spatial). This helper converts
    them to (B, *spatial) and ensures dtype is `torch.long`.

    Parameters
    ----------
    target:
        Target tensor of shape (B, 1, *spatial) or (B, *spatial).

    Returns
    -------
    torch.Tensor
        Label tensor of shape (B, *spatial) with dtype `torch.long`.
    """
    if target.ndim >= 2 and target.shape[1] == 1:
        return target[:, 0].long()
    return target.long()


def _foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    """Compute a foreground probability map from network logits.

    - If `logits` has one channel (C=1): assumes sigmoid-style binary training
      and returns `sigmoid(logits)`.
    - If `logits` has multiple channels (C>=2): assumes softmax-style training
      and returns the probability of "any foreground", computed as the union of
      all non-background classes: sum_{c=1..C-1} softmax(logits)[c].

    Parameters
    ----------
    logits:
        Network output logits of shape (B, C, *spatial).

    Returns
    -------
    torch.Tensor
        Foreground probability of shape (B, 1, *spatial).
    """
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)

    probs = torch.softmax(logits, dim=1)
    fg = probs[:, 1:].sum(dim=1, keepdim=True)
    return fg


class ComponentAdaptiveTverskyLoss(nn.Module):
    """Component-adaptive (lesion-balanced) Tversky loss for binary targets.

    This loss reweights **foreground voxels** inversely by the size of their
    connected component in the *ground-truth* mask. Intuition: each lesion
    (connected component) should contribute more equally to the objective,
    preventing large lesions from dominating the gradients.

    Implementation notes
    --------------------
    - Expects a binary foreground vs background target (labels > 0 are treated
      as foreground).
    - Works with nnUNet softmax outputs by converting logits to a single
      foreground-union probability map.
    - Requires SciPy (`scipy.ndimage`) for connected-component labeling.

    Parameters
    ----------
    alpha, beta:
        Tversky trade-off parameters (FP vs FN).
    gamma:
        Component size exponent. Larger values put more emphasis on small
        lesions.
    eps_cc:
        Small constant added to component size to stabilize very tiny lesions.
    w_bg:
        Background voxel weight (kept small to avoid background dominating).
    smooth:
        Numerical stability constant.
    connectivity:
        Connectivity for connected components (1 is 6-neighborhood in 3D).
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        eps_cc: float = 5.0,
        w_bg: float = 0.1,
        smooth: float = 1e-5,
        connectivity: int = 1,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps_cc = float(eps_cc)
        self.w_bg = float(w_bg)
        self.smooth = float(smooth)
        self.connectivity = int(connectivity)

    def _weight_map(self, gt_fg: torch.Tensor) -> torch.Tensor:
        """Build per-voxel weights based on GT connected components."""
        try:
            import scipy.ndimage as ndi
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SciPy is required for ComponentAdaptiveTverskyLoss. "
                "Please install scipy or implement a torch CC routine."
            ) from e

        gt_cpu = gt_fg.detach().cpu().numpy().astype(np.uint8)
        batch_weights = []

        if gt_cpu.ndim == 3:
            gt_cpu = gt_cpu[None]

        for b in range(gt_cpu.shape[0]):
            struct = ndi.generate_binary_structure(
                gt_cpu[b].ndim,
                self.connectivity,
            )
            lab, _ = ndi.label(gt_cpu[b], structure=struct)
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0

            denom = (sizes + self.eps_cc).astype(np.float32)
            inv = np.zeros_like(denom, dtype=np.float32)
            nonzero = denom > 0
            inv[nonzero] = denom[nonzero] ** (-self.gamma)

            w_fg = inv[lab].astype(np.float32)
            w = np.where(gt_cpu[b] > 0, w_fg, self.w_bg).astype(np.float32)
            batch_weights.append(w)

        w_np = np.stack(batch_weights, axis=0)
        w_t = torch.from_numpy(w_np).to(gt_fg.device)
        return w_t

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        fg_prob = _foreground_probability(net_output)
        tgt = _safe_squeeze_target(target)
        gt_fg = (tgt > 0).float()

        if gt_fg.sum() == 0:
            # No foreground in this patch: avoid unstable CC labeling and
            # return a zero loss contribution.
            return fg_prob.new_tensor(0.0)

        w = self._weight_map(gt_fg).unsqueeze(1)

        spatial_dims = list(range(2, fg_prob.ndim))
        gt = gt_fg.unsqueeze(1)
        tp = (w * fg_prob * gt).sum(dim=spatial_dims)
        fp = (w * fg_prob * (1.0 - gt)).sum(dim=spatial_dims)
        fn = (w * (1.0 - fg_prob) * gt).sum(dim=spatial_dims)

        num = tp + self.smooth
        den = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = num / den
        return 1.0 - tversky.mean()


class ComponentAdaptiveTverskyLossMultiClass(nn.Module):
    """Component-adaptive Tversky loss for multi-class segmentation.

    This variant applies a component-adaptive weighting **per foreground class**
    (excluding background class 0) and averages the loss across the classes
    that are present in the current patch.

    Notes
    -----
    - Expects softmax-style multi-class logits (C>=2). For binary (C=1), it
      falls back to a single foreground channel.
    - Component weights are computed from *ground-truth* connected components
      **within each class mask**.
    - Uses SciPy on CPU for connected-component labeling.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        eps_cc: float = 5.0,
        w_bg: float = 0.1,
        smooth: float = 1e-5,
        connectivity: int = 1,
        class_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eps_cc = float(eps_cc)
        self.w_bg = float(w_bg)
        self.smooth = float(smooth)
        self.connectivity = int(connectivity)
        self.class_ids = class_ids

    def _weight_map(self, gt_fg: torch.Tensor) -> torch.Tensor:
        """Build per-voxel weights based on GT connected components."""
        try:
            import scipy.ndimage as ndi
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SciPy is required for ComponentAdaptiveTverskyLossMultiClass. "
                "Please install scipy or implement a torch CC routine."
            ) from e

        gt_cpu = gt_fg.detach().cpu().numpy().astype(np.uint8)
        batch_weights = []

        if gt_cpu.ndim == 3:
            gt_cpu = gt_cpu[None]

        for b in range(gt_cpu.shape[0]):
            struct = ndi.generate_binary_structure(
                gt_cpu[b].ndim,
                self.connectivity,
            )
            lab, _ = ndi.label(gt_cpu[b], structure=struct)
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0

            denom = (sizes + self.eps_cc).astype(np.float32)
            inv = np.zeros_like(denom, dtype=np.float32)
            nonzero = denom > 0
            inv[nonzero] = denom[nonzero] ** (-self.gamma)

            w_fg = inv[lab].astype(np.float32)
            w = np.where(gt_cpu[b] > 0, w_fg, self.w_bg).astype(np.float32)
            batch_weights.append(w)

        w_np = np.stack(batch_weights, axis=0)
        w_t = torch.from_numpy(w_np).to(gt_fg.device)
        return w_t

    def _tversky_from_prob(
        self,
        prob: torch.Tensor,
        gt_fg: torch.Tensor,
    ) -> torch.Tensor:
        """Compute component-adaptive Tversky from a prob map and fg mask."""
        if gt_fg.sum() == 0:
            return prob.new_tensor(0.0)

        w = self._weight_map(gt_fg).unsqueeze(1)
        gt = gt_fg.unsqueeze(1)
        spatial_dims = list(range(2, prob.ndim))

        tp = (w * prob * gt).sum(dim=spatial_dims)
        fp = (w * prob * (1.0 - gt)).sum(dim=spatial_dims)
        fn = (w * (1.0 - prob) * gt).sum(dim=spatial_dims)

        num = tp + self.smooth
        den = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = num / den
        return 1.0 - tversky.mean()

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        tgt = _safe_squeeze_target(target)

        if net_output.shape[1] == 1:
            prob = torch.sigmoid(net_output)
            gt_fg = (tgt > 0).float()
            return self._tversky_from_prob(prob, gt_fg)

        probs = torch.softmax(net_output, dim=1)
        n_classes = int(probs.shape[1])

        if self.class_ids is None:
            class_ids = list(range(1, n_classes))
        else:
            class_ids = [
                int(c)
                for c in self.class_ids
                if 0 <= int(c) < n_classes
            ]

        losses: list[torch.Tensor] = []
        for c in class_ids:
            gt_c = (tgt == c).float()
            if gt_c.sum() == 0:
                continue
            prob_c = probs[:, c : c + 1]
            losses.append(self._tversky_from_prob(prob_c, gt_c))

        if len(losses) == 0:
            return probs.new_tensor(0.0)

        return torch.stack(losses).mean()

# ---- nnUNetTrainerMIL.py
class SmallLesionMILLoss(nn.Module):
    """Lesion-level MIL penalty to reduce missed small lesions.

    This auxiliary term treats each connected component in the ground-truth
    foreground as a "bag" (Multiple Instance Learning). A lesion is considered
    detected if at least one voxel within that component has high predicted
    probability.

    For each connected component c, we penalize low max-probability inside it:
        -log(max_{v in c} p(v)).

    Notes
    -----
    - Expects a binary foreground vs background target (labels > 0 are treated
      as foreground).
    - Works with nnUNet softmax outputs by converting logits to a single
      foreground-union probability map.
    - Requires SciPy (`scipy.ndimage`) for connected-component labeling.
    - Designed to be added on top of the default nnUNet loss with a small
      weight (lambda_mil).
    """

    def __init__(
        self,
        eps: float = 1e-6,
        connectivity: int = 1,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.connectivity = int(connectivity)

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        try:
            import scipy.ndimage as ndi
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SciPy is required for SmallLesionMILLoss. "
                "Please install scipy or implement a torch CC routine."
            ) from e

        fg_prob = _foreground_probability(net_output)
        tgt = _safe_squeeze_target(target)
        gt_fg = (tgt > 0).float()

        if gt_fg.sum() == 0:
            # No foreground in this patch: no lesion-level MIL signal.
            return fg_prob.new_tensor(0.0)

        # Connected components are computed from GT on CPU (constant wrt model),
        # but the MIL score (max prob inside each component) must be computed
        # on the *non-detached* probability map to keep gradients.
        gt_cpu = gt_fg.detach().cpu().numpy().astype(np.uint8)

        if gt_cpu.ndim == 3:
            gt_cpu = gt_cpu[None]

        losses_t = []
        for b in range(gt_cpu.shape[0]):
            struct = ndi.generate_binary_structure(
                gt_cpu[b].ndim,
                self.connectivity,
            )
            lab_np, num = ndi.label(gt_cpu[b], structure=struct)
            if num == 0:
                continue

            lab = torch.from_numpy(lab_np).to(
                device=fg_prob.device,
                dtype=torch.int64,
            )

            for k in range(1, num + 1):
                mask = lab == k
                if not bool(mask.any()):
                    continue

                # Differentiable max over the component.
                m = fg_prob[b, 0][mask].max()
                m = torch.clamp(m, min=self.eps)
                losses_t.append(-torch.log(m))

        if len(losses_t) == 0:
            return fg_prob.new_tensor(0.0)

        return torch.stack(losses_t, dim=0).mean()

# ---- nnUNetTrainerFocalTversky.py / nnUNetTrainerTversky.py (MONAI)
from monai.losses import TverskyLoss

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float, beta: float, gamma: float):
        super().__init__()
        self.gamma = gamma
        self.tversky_loss = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            alpha=alpha,
            beta=beta,
            reduction="mean",
        )

    def forward(self, y_hat, y):
        tversky_loss = self.tversky_loss(y_hat, y)
        return torch.pow(tversky_loss, 1.0 / self.gamma)

class MyTverskyLoss(torch.nn.Module):
    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.tversky_loss = TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            alpha=alpha,
            beta=beta,
        )

    def forward(self, y_hat, y):
        return self.tversky_loss(y_hat, y)

# ---- nnUNetTrainerCCDiceCE.py (ccdicece-baseline)
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


