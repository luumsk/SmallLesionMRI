from typing import Union, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper



def _safe_squeeze_target(target: torch.Tensor) -> torch.Tensor:
    """Ensure target is (B, *spatial) with integer labels."""
    if target.ndim >= 2 and target.shape[1] == 1:
        return target[:, 0].long()
    return target.long()


def _foreground_probability(logits: torch.Tensor) -> torch.Tensor:
    """Return foreground probability map of shape (B, 1, *spatial).

    For softmax training (C>=2), foreground is the union of all non-background
    classes. For sigmoid/regions training, this returns sigmoid(logits).
    """
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    probs = torch.softmax(logits, dim=1)
    fg = probs[:, 1:].sum(dim=1, keepdim=True)
    return fg


class ComponentAdaptiveTverskyLoss(nn.Module):
    """Component-adaptive (lesion-balanced) Tversky loss.

    This reweights foreground voxels inversely by their connected-component size
    so that small lesions contribute more to the loss.

    Notes
    -----
    - Designed for binary foreground vs background training.
    - Works with nnUNet softmax outputs by using foreground union prob.
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
            lab, num = ndi.label(gt_cpu[b],
                                structure=ndi.generate_binary_structure(
                                    gt_cpu[b].ndim, self.connectivity))
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

        w = self._weight_map(gt_fg)
        w = w.unsqueeze(1)

        tp = (w * fg_prob * gt_fg.unsqueeze(1)).sum(dim=list(range(2,
                                                                   fg_prob.ndim)))
        fp = (w * fg_prob * (1.0 - gt_fg.unsqueeze(1))).sum(
            dim=list(range(2, fg_prob.ndim))
        )
        fn = (w * (1.0 - fg_prob) * gt_fg.unsqueeze(1)).sum(
            dim=list(range(2, fg_prob.ndim))
        )

        num = tp + self.smooth
        den = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = num / den
        return 1.0 - tversky.mean()


class SmallLesionMILLoss(nn.Module):
    """Lesion-level MIL penalty: each GT lesion should have a confident voxel.

    For each connected component c in the GT foreground, penalize low
    max-probability within that component: -log(max_{v in c} p(v)).

    This term is small and should be used as an auxiliary loss.
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

        gt_cpu = gt_fg.detach().cpu().numpy().astype(np.uint8)
        p_cpu = fg_prob.detach().cpu().numpy().astype(np.float32)[:, 0]

        if gt_cpu.ndim == 3:
            gt_cpu = gt_cpu[None]
            p_cpu = p_cpu[None]

        losses = []
        for b in range(gt_cpu.shape[0]):
            lab, num = ndi.label(gt_cpu[b],
                                structure=ndi.generate_binary_structure(
                                    gt_cpu[b].ndim, self.connectivity))
            if num == 0:
                continue
            for k in range(1, num + 1):
                mask = lab == k
                if not np.any(mask):
                    continue
                m = float(p_cpu[b][mask].max())
                losses.append(-np.log(max(m, self.eps)))

        if len(losses) == 0:
            return fg_prob.new_tensor(0.0)

        return fg_prob.new_tensor(float(np.mean(losses)))


class CATAndMILLoss(nn.Module):
    """Combined CAT (component-adaptive Tversky) + MIL term."""

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        eps_cc: float = 5.0,
        w_bg: float = 0.1,
        smooth: float = 1e-5,
        lambda_mil: float = 0.2,
        connectivity: int = 1,
    ) -> None:
        super().__init__()
        self.cat = ComponentAdaptiveTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            eps_cc=eps_cc,
            w_bg=w_bg,
            smooth=smooth,
            connectivity=connectivity,
        )
        self.mil = SmallLesionMILLoss(
            eps=1e-6,
            connectivity=connectivity,
        )
        self.lambda_mil = float(lambda_mil)

    def forward(
        self,
        net_output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.cat(net_output, target) + self.lambda_mil * self.mil(
            net_output, target
        )



class nnUNetTrainerCAT(nnUNetTrainer):
    """Simplest CAT trainer for debugging.

    This version keeps the CPU connected-component computation inside the loss
    to minimize changes elsewhere. Use this first to confirm there are no
    training/runtime errors.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda'),
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        eps_cc: float = 5.0,
        w_bg: float = 0.1,
        connectivity: int = 1,
    ) -> None:
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )
        self._cat_params = {
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma),
            'eps_cc': float(eps_cc),
            'w_bg': float(w_bg),
            'connectivity': int(connectivity),
        }

    def _build_loss(self):
        base = ComponentAdaptiveTverskyLoss(**self._cat_params)
        if getattr(self, 'enable_deep_supervision', False):
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            return DeepSupervisionWrapper(base, weights)
        return base


class nnUNetTrainerMIL(nnUNetTrainer):
    """Simplest MIL-only trainer for debugging.

    Uses the default nnUNet loss and adds a small MIL penalty computed on the
    highest-resolution output (output[0]).
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda'),
        lambda_mil: float = 0.2,
        connectivity: int = 1,
    ) -> None:
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )
        self.lambda_mil = float(lambda_mil)
        self.connectivity = int(connectivity)

    def _build_loss(self):
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

                return self.base_loss(net_output, target) + self.lambda_mil * \
                    self.mil_loss(out, tgt)

        return _MILWrappedLoss(base, mil, self.lambda_mil)


class nnUNetTrainerCATMIL(nnUNetTrainer):
    """Simplest combined CAT + MIL trainer for debugging."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device('cuda'),
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        eps_cc: float = 5.0,
        w_bg: float = 0.1,
        lambda_mil: float = 0.2,
        connectivity: int = 1,
    ) -> None:
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            unpack_dataset=unpack_dataset,
            device=device,
        )
        self._catmil_params = {
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma),
            'eps_cc': float(eps_cc),
            'w_bg': float(w_bg),
            'lambda_mil': float(lambda_mil),
            'connectivity': int(connectivity),
        }

    def _build_loss(self):
        base = CATAndMILLoss(**self._catmil_params)
        if getattr(self, 'enable_deep_supervision', False):
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
            weights[-1] = 0
            weights = weights / weights.sum()
            return DeepSupervisionWrapper(base, weights)
        return base