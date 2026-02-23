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


class nnUNetTrainerMIL(nnUNetTrainer):
    """nnUNet trainer variant using MIL as an auxiliary lesion-detection term.

    Intended for **ablation/debugging** first.

    This trainer keeps the default nnUNet loss (typically Dice+CE with deep
    supervision) and adds :class:`SmallLesionMILLoss` computed on the
    highest-resolution output (output[0]).

    The MIL term uses CPU SciPy connected-components for simplicity; once the
    concept is validated, an optimized variant can compute CC information in
    the CPU data pipeline to avoid GPU->CPU transfers.
    """

    # NOTE:
    # nnUNet v2 stores trainer constructor arguments in `self.my_init_kwargs`.
    # The base `nnUNetTrainer.__init__` populates that dict from *its own*
    # `locals()`. If this subclass adds extra `__init__` parameters (e.g.,
    # lambda_mil/connectivity), the base class may attempt to read them and
    # crash with a KeyError. Therefore, keep the same signature as the base
    # trainer and store MIL hyperparameters as attributes.

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

        # MIL hyperparameters
        lambda_mil = 0.2
        connectivity = 1

        self.lambda_mil = float(lambda_mil)
        self.connectivity = int(connectivity)

    def _build_loss(self) -> nn.Module:
        base = super()._build_loss()
        mil = SmallLesionMILLoss(connectivity=self.connectivity)

        class _MILWrappedLoss(nn.Module):
            def __init__(
                self,
                base_loss: nn.Module,
                mil_loss: nn.Module,
                lambda_mil_: float,
            ) -> None:
                super().__init__()
                self.base_loss = base_loss
                self.mil_loss = mil_loss
                self.lambda_mil = float(lambda_mil_)

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
                    + self.lambda_mil * self.mil_loss(out, tgt)
                )

        return _MILWrappedLoss(base, mil, self.lambda_mil)
