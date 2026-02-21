from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerCAT import (
    ComponentAdaptiveTverskyLoss,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMIL import SmallLesionMILLoss



class CATAndMILLoss(nn.Module):
    """Combined CAT + MIL loss for small lesion sensitivity.

    This combines:
    - ComponentAdaptiveTverskyLoss (CAT): lesion-balanced overlap objective
    - SmallLesionMILLoss (MIL): lesion-level detection surrogate

    Notes
    -----
    - The underlying CAT and MIL implementations rely on SciPy connected
      components inside their loss computations (CPU). This is intended as a
      simplest/ablation implementation. After validation, you can optimize by
      computing component information in the CPU data pipeline.
    """

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
        return (
            self.cat(net_output, target)
            + self.lambda_mil * self.mil(net_output, target)
        )


class nnUNetTrainerCATMIL(nnUNetTrainer):
    """nnUNet trainer variant using combined CAT + MIL loss.

    Intended for **ablation/debugging** first.

    This trainer keeps the default nnUNet loss and adds CAT and MIL auxiliaries.
    Deep supervision is supported via :class:`DeepSupervisionWrapper`.

    After correctness is confirmed, consider an optimized variant that computes
    CAT weights / CC information in the CPU data pipeline to avoid GPU->CPU
    transfers inside the loss.
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
        lambda_cat: float = 0.3,
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
        self.lambda_cat = float(lambda_cat)
        self._catmil_params = {
            'alpha': float(alpha),
            'beta': float(beta),
            'gamma': float(gamma),
            'eps_cc': float(eps_cc),
            'w_bg': float(w_bg),
            'lambda_cat': float(lambda_cat),
            'lambda_mil': float(lambda_mil),
            'connectivity': int(connectivity),
        }

    def _build_loss(self) -> nn.Module:
        base = super()._build_loss()
        cat = ComponentAdaptiveTverskyLoss(
            alpha=self._catmil_params['alpha'],
            beta=self._catmil_params['beta'],
            gamma=self._catmil_params['gamma'],
            eps_cc=self._catmil_params['eps_cc'],
            w_bg=self._catmil_params['w_bg'],
            smooth=1e-5,
            connectivity=self._catmil_params['connectivity'],
        )
        mil = SmallLesionMILLoss(
            eps=1e-6,
            connectivity=self._catmil_params['connectivity'],
        )
        lambda_cat = float(self._catmil_params['lambda_cat'])
        lambda_mil = float(self._catmil_params['lambda_mil'])

        class _CATMILWrappedLoss(nn.Module):
            def __init__(
                self,
                base_loss: nn.Module,
                cat_loss: nn.Module,
                mil_loss: nn.Module,
                lambda_cat_: float,
                lambda_mil_: float,
            ) -> None:
                super().__init__()
                self.base_loss = base_loss
                self.cat_loss = cat_loss
                self.mil_loss = mil_loss
                self.lambda_cat = float(lambda_cat_)
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
                    + self.lambda_cat * self.cat_loss(out, tgt)
                    + self.lambda_mil * self.mil_loss(out, tgt)
                )

        return _CATMILWrappedLoss(
            base,
            cat,
            mil,
            lambda_cat,
            lambda_mil,
        )