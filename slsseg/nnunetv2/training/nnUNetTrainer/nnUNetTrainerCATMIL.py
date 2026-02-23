from __future__ import annotations

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerCAT import (
    ComponentAdaptiveTverskyLoss,
)
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerMIL import SmallLesionMILLoss


class nnUNetTrainerCATMIL(nnUNetTrainer):
    """nnUNet trainer variant using combined CAT + MIL loss.

    Intended for **ablation/debugging** first.

    This trainer keeps the default nnUNet loss and adds CAT and MIL auxiliaries.
    Deep supervision is supported via :class:`DeepSupervisionWrapper`.

    After correctness is confirmed, consider an optimized variant that computes
    CAT weights / CC information in the CPU data pipeline to avoid GPU->CPU
    transfers inside the loss.
    """

    # NOTE:
    # nnUNet v2 stores trainer constructor arguments in `self.my_init_kwargs`.
    # The base `nnUNetTrainer.__init__` populates that dict from *its own*
    # `locals()`. If this subclass adds extra `__init__` parameters (e.g.,
    # alpha/beta/lambda_*), the base class may attempt to read them and crash
    # with a KeyError. Therefore, keep the same signature as the base trainer
    # and store CAT/MIL hyperparameters as attributes.

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

        # CAT + MIL hyperparameters
        alpha = 0.3
        beta = 0.7
        gamma = 1.0
        eps_cc = 5.0
        w_bg = 0.01
        lambda_cat = 0.1
        lambda_mil = 0.2
        connectivity = 1

        self.lambda_cat = float(lambda_cat)
        self.lambda_cat_final = float(lambda_cat)
        self.lambda_cat_warmup_epochs = 50
        self.lambda_cat = 0.0

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

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()

        warmup = int(self.lambda_cat_warmup_epochs)
        if warmup <= 0:
            self.lambda_cat = float(self.lambda_cat_final)
            return

        # nnUNet uses 0-based epochs.
        t = min(max(self.current_epoch, 0), warmup) / float(warmup)
        self.lambda_cat = float(t * self.lambda_cat_final)

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
        lambda_mil = float(self._catmil_params['lambda_mil'])

        class _CATMILWrappedLoss(nn.Module):
            def __init__(
                self,
                base_loss: nn.Module,
                cat_loss: nn.Module,
                mil_loss: nn.Module,
                trainer_ref: nnUNetTrainer,
                lambda_mil_: float,
            ) -> None:
                super().__init__()
                self.base_loss = base_loss
                self.cat_loss = cat_loss
                self.mil_loss = mil_loss
                self.trainer = trainer_ref
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
                    + float(self.trainer.lambda_cat) * self.cat_loss(out, tgt)
                    + self.lambda_mil * self.mil_loss(out, tgt)
                )

        return _CATMILWrappedLoss(
            base,
            cat,
            mil,
            self,
            lambda_mil,
        )