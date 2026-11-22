import numpy as np
import pytest
import torch
from scipy import ndimage

import catmil.cc


def make_batch(b=2, c=2, spatial=(24, 24, 24), p_lesion=0.004, seed=0, n_classes=None):
    """Random logits and a label map with small blobs of several sizes."""
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(b, c, *spatial, generator=g)
    rng = np.random.default_rng(seed)
    seeds = rng.random((b, *spatial)) < p_lesion
    # grow about half of the seeds by one voxel -> lesions of 1 to ~10 voxels
    # (iterations=0 would mean 'dilate until convergence' in scipy, so never pass 0)
    big = seeds & (rng.random(seeds.shape) < 0.5)
    grown = np.stack([s | ndimage.binary_dilation(bg, iterations=1) for s, bg in zip(seeds, big)])
    lab = grown.astype(np.int64)
    k = n_classes if n_classes is not None else max(c, 2)
    if k > 2:  # assign each blob a random class 1..k-1
        for i in range(b):
            cc, n = ndimage.label(grown[i])
            cls = rng.integers(1, k, size=n + 1)
            cls[0] = 0
            lab[i] = cls[cc]
    return logits, torch.from_numpy(lab[:, None]).float()


@pytest.fixture(params=["scipy", "cc3d"] if catmil.cc._HAS_CC3D else ["scipy"])
def cc_backend(request):
    old = catmil.cc.BACKEND
    catmil.cc.set_backend(request.param)
    yield request.param
    catmil.cc.BACKEND = old
