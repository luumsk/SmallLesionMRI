from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

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
    """Component-adaptive (lesion-balanced) Tversky loss for small lesions.

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

        w = self._weight_map(gt_fg)
        w = w.unsqueeze(1)

        spatial_dims = list(range(2, fg_prob.ndim))
        tp = (w * fg_prob * gt_fg.unsqueeze(1)).sum(dim=spatial_dims)
        fp = (w * fg_prob * (1.0 - gt_fg.unsqueeze(1))).sum(dim=spatial_dims)
        fn = (w * (1.0 - fg_prob) * gt_fg.unsqueeze(1)).sum(dim=spatial_dims)

        num = tp + self.smooth
        den = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = num / den
        return 1.0 - tversky.mean()


class nnUNetTrainerCAT(nnUNetTrainer):
    """nnUNet trainer variant using component-adaptive Tversky (CAT).

    Intended for **ablation/debugging** first.

    This trainer keeps the default nnUNet loss (typically Dice+CE with deep
    supervision) and adds :class:`ComponentAdaptiveTverskyLoss` as an
    auxiliary term computed on the highest-resolution output (output[0]).
    The connected-component computation is performed inside the loss via a
    CPU SciPy call for simplicity.
    """

    # NOTE:
    # nnUNet v2 stores trainer constructor arguments in `self.my_init_kwargs`.
    # The base `nnUNetTrainer.__init__` populates that dict from *its own*
    # `locals()`. If this subclass adds extra `__init__` parameters (e.g.,
    # alpha/beta), the base class will attempt to read them and crash with a
    # KeyError. Therefore, keep the same signature as the base trainer and
    # store CAT hyperparameters as attributes.

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda'),
    ) -> None:
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )

        # CAT hyperparameters
        alpha = 0.3
        beta = 0.7
        gamma = 1.0
        eps_cc = 5.0
        w_bg = 0.1
        connectivity = 1
        lambda_cat = 0.3

        self._cat_params = {
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma),
            'eps_cc': float(eps_cc),
            'w_bg': float(w_bg),
            'connectivity': int(connectivity),
        }
        self.lambda_cat = float(lambda_cat)

    def _build_loss(self) -> nn.Module:
        base = super()._build_loss()
        cat = ComponentAdaptiveTverskyLoss(**self._cat_params)

        class _CATWrappedLoss(nn.Module):
            def __init__(
                self,
                base_loss: nn.Module,
                cat_loss: nn.Module,
                lambda_cat_: float,
            ) -> None:
                super().__init__()
                self.base_loss = base_loss
                self.cat_loss = cat_loss
                self.lambda_cat = float(lambda_cat_)

            def forward(self, net_output, target) -> torch.Tensor:
                if isinstance(net_output, (list, tuple)):
                    out = net_output[0]
                else:
                    out = net_output

                if isinstance(target, (list, tuple)):
                    tgt = target[0]
                else:
                    tgt = target

                return (
                    self.base_loss(net_output, target)
                    + self.lambda_cat * self.cat_loss(out, tgt)
                )

        return _CATWrappedLoss(base, cat, self.lambda_cat)