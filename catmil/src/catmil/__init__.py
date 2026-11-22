"""catmil: losses for small lesion segmentation (CATMIL and the baselines of the paper).

Quick start::

    from catmil import CATMILLoss
    loss_fn = CATMILLoss()            # DiceCE + 0.1 * CAT (50-epoch warm-up) + 0.2 * MIL
    for epoch in range(n_epochs):
        loss_fn.set_epoch(epoch)
        ...
        loss = loss_fn(logits, labels)  # logits (B, C, *spatial), labels (B, 1, *spatial)
"""
from .cc import label_components, set_backend
from .losses import *  # noqa: F401,F403
from .losses import __all__ as _losses_all
from .transforms import LabelComponents, LabelComponentsd

__version__ = "0.1.0"
__all__ = list(_losses_all) + ["LabelComponents", "LabelComponentsd", "label_components", "set_backend"]
