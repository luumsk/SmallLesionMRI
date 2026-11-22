"""The package losses must reproduce the research implementations (values and gradients)."""
import pytest
import torch

import _reference_losses as ref
from catmil import (CCDiceCELoss, ComponentAdaptiveTverskyLoss, FocalTverskyLoss, SmallLesionMILLoss,
                    TverskyLoss)
from conftest import make_batch

TOL = dict(rtol=1e-5, atol=1e-6)


def _val_grad(loss_fn, logits, target):
    x = logits.clone().requires_grad_(True)
    v = loss_fn(x, target)
    v.backward()
    return v.detach(), x.grad


def _check(new, old, logits, target):
    v1, g1 = _val_grad(new, logits, target)
    v0, g0 = _val_grad(old, logits, target)
    torch.testing.assert_close(v1, v0.to(v1.dtype), **TOL)
    torch.testing.assert_close(g1, g0, **TOL)


SHAPES = [(2, (24, 24, 24)), (3, (40, 40))]


class _Lift2D(torch.nn.Module):
    """The research losses label (B, H, W) as one 3D volume (2D bug), so compare 2D inputs
    against the reference on depth-1 3D volumes, which is the correct per-sample 2D result."""

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, x, y):
        return self.loss(x.unsqueeze(2), y.unsqueeze(2)) if x.ndim == 4 else self.loss(x, y)


@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("b,spatial", SHAPES)
def test_cat_binary(cc_backend, c, b, spatial):
    logits, target = make_batch(b, c, spatial, seed=1)
    kw = dict(alpha=0.3, beta=0.7, gamma=1.0, eps_cc=5.0, w_bg=0.01, smooth=1e-5, connectivity=1)
    _check(ComponentAdaptiveTverskyLoss(**kw), _Lift2D(ref.ComponentAdaptiveTverskyLoss(**kw)), logits, target)


@pytest.mark.parametrize("gamma,conn", [(0.5, 1), (2.0, 3)])
def test_cat_hyperparameters(gamma, conn):
    logits, target = make_batch(2, 2, (20, 20, 20), seed=2)
    kw = dict(gamma=gamma, w_bg=0.05, eps_cc=1.0, connectivity=conn)
    _check(ComponentAdaptiveTverskyLoss(**kw), ref.ComponentAdaptiveTverskyLoss(**kw), logits, target)


def test_cat_multiclass(cc_backend):
    logits, target = make_batch(2, 4, (24, 24, 24), seed=3, p_lesion=0.01)
    kw = dict(w_bg=0.01)
    _check(ComponentAdaptiveTverskyLoss(mode="per_class", **kw),
           ref.ComponentAdaptiveTverskyLossMultiClass(**kw), logits, target)
    _check(ComponentAdaptiveTverskyLoss(mode="auto", **kw),
           ref.ComponentAdaptiveTverskyLossMultiClass(**kw), logits, target)


@pytest.mark.parametrize("c", [1, 2])
@pytest.mark.parametrize("b,spatial", SHAPES)
def test_mil(cc_backend, c, b, spatial):
    logits, target = make_batch(b, c, spatial, seed=4)
    _check(SmallLesionMILLoss(), _Lift2D(ref.SmallLesionMILLoss()), logits, target)


def test_tversky_matches_monai():
    logits, target = make_batch(2, 2, (16, 16, 16), seed=5)
    _check(TverskyLoss(0.3, 0.7), ref.MyTverskyLoss(0.3, 0.7), logits, target)
    logits, target = make_batch(2, 3, (16, 16, 16), seed=6, p_lesion=0.02)
    _check(TverskyLoss(0.3, 0.7), ref.MyTverskyLoss(0.3, 0.7), logits, target)


def test_focal_tversky():
    logits, target = make_batch(2, 2, (16, 16, 16), seed=7)
    _check(FocalTverskyLoss(0.3, 0.7, 4 / 3), ref.FocalTverskyLoss(0.3, 0.7, 4 / 3), logits, target)


@pytest.mark.parametrize("shape", [(32, 32, 32), (16, 16, 16), (8, 8, 8)])
def test_cc_dice_ce(shape):
    """Full resolution and two deep-supervision levels with anisotropic spacing."""
    logits, target = make_batch(2, 2, shape, seed=8, p_lesion=0.01)
    kw = dict(spacing=(1.0, 0.8, 3.0), patch_size=(32, 32, 32))
    _check(CCDiceCELoss(**kw), ref.CCDiceCELoss(**kw), logits, target)


def test_cc_dice_ce_empty_sample():
    logits, target = make_batch(2, 2, (16, 16, 16), seed=9)
    target[1] = 0
    kw = dict(spacing=(1, 1, 1), patch_size=(16, 16, 16))
    _check(CCDiceCELoss(**kw), ref.CCDiceCELoss(**kw), logits, target)
