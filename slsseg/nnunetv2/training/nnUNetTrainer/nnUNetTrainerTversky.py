from monai.losses import TverskyLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import numpy as np
import torch


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


class nnUNetTrainerTversky(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        
        """nnU-Net trainer using Tversky loss."""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        self.initial_lr = 1e-2
        self.num_epochs = 150
        self.save_every = 5
        self.disable_checkpointing = False
    
    def _build_loss(self):
        loss = MyTverskyLoss(alpha=0.3, beta=0.7)

        deep_supervision_scales = self._get_deep_supervision_scales()

        # we give each output a weight which decreases exponentially (division by 2) as the resolution decreases
        # this gives higher resolution outputs more weight in the loss
        weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
        weights[-1] = 0

        # we don't use the lowest 2 outputs. Normalize weights so that they sum to 1
        weights = weights / weights.sum()
        # now wrap the loss
        loss = DeepSupervisionWrapper(loss, weights)

        return loss
