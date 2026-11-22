"""Name -> loss lookup, so experiments can switch losses from a config string."""
from __future__ import annotations

import re
from typing import Callable, Dict, List

import torch.nn as nn

from .baselines import CCDiceCEGlobalLoss, CCDiceCELoss, DiceCELoss, FocalTverskyLoss, TverskyLoss
from .cat import ComponentAdaptiveTverskyLoss
from .catmil import CATMILLoss
from .deep_supervision import DeepSupervisionLoss
from .mil import SmallLesionMILLoss

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "dice_ce": DiceCELoss,
    "tversky": TverskyLoss,
    "focal_tversky": FocalTverskyLoss,
    "cc_dice_ce": CCDiceCEGlobalLoss,          # the paper's CC-DiceCE objective (CC + global DiceCE)
    "cc_dice_ce_component": CCDiceCELoss,      # component term only
    "cat": ComponentAdaptiveTverskyLoss,       # auxiliary term alone
    "mil": SmallLesionMILLoss,                 # auxiliary term alone
    "catmil": CATMILLoss,                      # DiceCE + CAT + MIL (handles deep supervision itself)
}
_HANDLES_DS = {"catmil"}


_ALIASES = {re.sub(r"[^a-z0-9]", "", k): k for k in _REGISTRY}  # "CC-DiceCE" -> "cc_dice_ce"


def _key(name: str) -> str:
    k = name.lower().replace("-", "_").replace(" ", "_")
    return k if k in _REGISTRY else _ALIASES.get(re.sub(r"[^a-z0-9]", "", name.lower()), k)


def list_losses() -> List[str]:
    """Names accepted by ``get_loss``."""
    return sorted(_REGISTRY)


def register_loss(name: str, factory: Callable[..., nn.Module]) -> None:
    """Add a custom loss under ``name`` (e.g. to compare it with the built-in ones)."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    _REGISTRY[key] = factory
    _ALIASES[re.sub(r"[^a-z0-9]", "", key)] = key


def get_loss(name: str, deep_supervision: bool = False, ds_weights=None, **kwargs) -> nn.Module:
    """Build a loss by name, e.g. ``get_loss("catmil", lambda_mil=0.2)``.

    ``deep_supervision=True`` wraps it in ``DeepSupervisionLoss`` (CATMIL handles lists itself).
    """
    key = _key(name)
    if key not in _REGISTRY:
        raise KeyError(f"unknown loss {name!r}; available: {', '.join(list_losses())}")
    if key in _HANDLES_DS:
        if ds_weights is not None:
            kwargs["ds_weights"] = ds_weights
        return _REGISTRY[key](**kwargs)
    loss = _REGISTRY[key](**kwargs)
    return DeepSupervisionLoss(loss, ds_weights) if deep_supervision else loss
