"""Connected-component labelling shared by the losses, transforms and metrics.

Connectivity follows ``scipy.ndimage.generate_binary_structure``: 1 = face neighbours
(6 in 3D, 4 in 2D), ``ndim`` = all neighbours (26 in 3D, 8 in 2D). The paper uses 1.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from scipy import ndimage

try:  # optional, faster labelling: pip install connected-components-3d
    import cc3d as _cc3d
    _HAS_CC3D = True
except Exception:  # pragma: no cover - depends on the environment
    _cc3d = None
    _HAS_CC3D = False

_CC3D_CONN = {(2, 1): 4, (2, 2): 8, (3, 1): 6, (3, 2): 18, (3, 3): 26}

BACKEND = "auto"  # "auto" (cc3d if installed, else scipy), "scipy" or "cc3d"


def set_backend(name: str) -> None:
    """Select the labelling backend for the whole process ("auto", "scipy", "cc3d")."""
    global BACKEND
    if name not in ("auto", "scipy", "cc3d"):
        raise ValueError(f"unknown backend {name!r}")
    if name == "cc3d" and not _HAS_CC3D:
        raise ImportError("cc3d backend requested but connected-components-3d is not installed")
    BACKEND = name


def label_components(mask, connectivity: int = 1) -> Tuple[np.ndarray, int]:
    """Label the connected components of a binary 2D or 3D mask.

    Parameters
    ----------
    mask : np.ndarray or torch.Tensor, shape (*spatial)
        Non-zero voxels are foreground.
    connectivity : int
        1 = faces only (paper default), up to ``mask.ndim``.

    Returns
    -------
    labels : np.ndarray of int64, same shape, 0 = background, 1..n = components
    n : int, number of components
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask) != 0
    use_cc3d = _HAS_CC3D and BACKEND in ("auto", "cc3d") and (mask.ndim, connectivity) in _CC3D_CONN
    if use_cc3d:
        lab, n = _cc3d.connected_components(
            mask.astype(np.uint8), connectivity=_CC3D_CONN[(mask.ndim, connectivity)], return_N=True)
        return lab.astype(np.int64, copy=False), int(n)
    struct = ndimage.generate_binary_structure(mask.ndim, connectivity)
    lab, n = ndimage.label(mask, structure=struct)
    return lab.astype(np.int64, copy=False), int(n)


def label_batch(mask: torch.Tensor, connectivity: int = 1) -> torch.Tensor:
    """Label each sample of a (B, *spatial) mask; returns int64 labels on ``mask.device``."""
    out = np.stack([label_components(m, connectivity)[0] for m in mask.detach().cpu().numpy()])
    return torch.from_numpy(out).to(mask.device)
