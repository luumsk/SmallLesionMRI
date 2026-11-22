"""Data-loader transforms that precompute GT connected components.

The CAT and MIL losses label the GT components in every call, which copies the target to the
CPU. Computing them in the data loader workers instead removes that sync from the training step;
pass the result as ``loss(logits, target, components=batch["label_cc"])``.

Only valid if no spatial augmentation runs after the transform (put it last), and only for the
full-resolution target (``ds_mode="highest"``).
"""
from __future__ import annotations

from typing import Hashable, Mapping, Optional

import numpy as np
import torch

from .cc import label_components


class LabelComponents:
    """Array transform: label map / mask (optionally with a leading channel axis of 1) -> int64 CCs.

    Foreground is ``label > 0`` (or ``label == class_id``); ``ignore_index`` voxels are excluded.
    """

    def __init__(self, connectivity: int = 1, class_id: Optional[int] = None,
                 ignore_index: Optional[int] = None) -> None:
        self.connectivity, self.class_id, self.ignore_index = int(connectivity), class_id, ignore_index

    def __call__(self, label):
        is_t = isinstance(label, torch.Tensor)
        arr = label.detach().cpu().numpy() if is_t else np.asarray(label)
        squeeze = arr.ndim >= 3 and arr.shape[0] == 1
        a = arr[0] if squeeze else arr
        fg = (a == self.class_id) if self.class_id is not None else (a > 0)
        if self.ignore_index is not None:
            fg &= a != self.ignore_index
        lab, _ = label_components(fg, self.connectivity)
        if squeeze:
            lab = lab[None]
        return torch.from_numpy(lab) if is_t else lab


class LabelComponentsd:
    """Dictionary version (MONAI style): ``data[out_key] = LabelComponents(data[key])``."""

    def __init__(self, key: Hashable = "label", out_key: Hashable = "label_cc", **kwargs) -> None:
        self.key, self.out_key = key, out_key
        self.t = LabelComponents(**kwargs)

    def __call__(self, data: Mapping) -> dict:
        d = dict(data)
        d[self.out_key] = self.t(d[self.key])
        return d
