"""Behaviour of the package API: edge cases, options, deep supervision, registry."""
import math
import warnings

import numpy as np
import pytest
import torch

import _reference_losses as ref
from catmil import (CATMILLoss, ComponentAdaptiveTverskyLoss, DeepSupervisionLoss, DiceCELoss,
                    LabelComponents, LabelComponentsd, SmallLesionMILLoss, get_loss, list_losses,
                    nnunet_ds_weights, register_loss)
from catmil.losses.mil import component_max
from conftest import make_batch


def test_empty_foreground_is_zero_and_finite():
    logits = torch.randn(2, 2, 8, 8, 8, requires_grad=True)
    target = torch.zeros(2, 1, 8, 8, 8)
    for fn in (ComponentAdaptiveTverskyLoss(), SmallLesionMILLoss()):
        v = fn(logits, target)
        assert float(v) == 0.0
    loss = CATMILLoss(cat_warmup_epochs=0)(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()


def test_single_voxel_lesions():
    logits = torch.full((1, 2, 9, 9, 9), -4.0)
    logits[:, 0] = 4.0  # confident background everywhere
    target = torch.zeros(1, 1, 9, 9, 9)
    target[0, 0, 2, 2, 2] = target[0, 0, 6, 6, 6] = 1
    mil = SmallLesionMILLoss()(logits, target)
    p = torch.softmax(torch.tensor([4.0, -4.0]), 0)[1]
    assert math.isclose(float(mil), float(-torch.log(p)), rel_tol=1e-5)


def test_component_max_matches_loop_and_skips_missing_ids():
    prob = torch.rand(6, 7, 8, requires_grad=True)
    lab = torch.zeros(6, 7, 8, dtype=torch.long)
    lab[:2] = 1
    lab[4:] = 5  # ids 2..4 absent
    m = component_max(prob, lab)
    assert m.shape == (2,)
    torch.testing.assert_close(m, torch.stack([prob[:2].max(), prob[4:].max()]))


def test_ignore_index_excluded():
    logits, target = make_batch(2, 2, (16, 16, 16), seed=11)
    t_ign = target.clone()
    t_ign[:, :, :4] = 2  # ignore a slab
    for cls in (ComponentAdaptiveTverskyLoss, SmallLesionMILLoss):
        v = cls(ignore_index=2)(logits, t_ign)
        assert torch.isfinite(v)
    # with nothing ignored, ignore_index must not change the value
    torch.testing.assert_close(ComponentAdaptiveTverskyLoss(ignore_index=2)(logits, target),
                               ComponentAdaptiveTverskyLoss()(logits, target))
    # an all-ignored patch has no lesions
    assert float(SmallLesionMILLoss(ignore_index=2)(logits, torch.full_like(target, 2))) == 0.0


def test_regions_sigmoid():
    logits = torch.randn(2, 3, 16, 16, 16, requires_grad=True)
    _, lab = make_batch(2, 2, (16, 16, 16), seed=12, p_lesion=0.01)
    regions = torch.cat([lab > 0] * 3, 1).float()
    for fn in (ComponentAdaptiveTverskyLoss(activation="sigmoid"), SmallLesionMILLoss(activation="sigmoid"),
               DiceCELoss(activation="sigmoid"), get_loss("tversky", activation="sigmoid"),
               CATMILLoss(activation="sigmoid", cat_warmup_epochs=0)):
        v = fn(logits, regions)
        v.backward()
        assert torch.isfinite(v)
    # region target + ignore channel (nnU-Net convention: last channel)
    ign = torch.zeros_like(regions[:, :1])
    ign[:, :, :3] = 1
    v = CATMILLoss(activation="sigmoid", ignore_index=1, cat_warmup_epochs=0)(logits, torch.cat([regions, ign], 1))
    assert torch.isfinite(v)


def test_precomputed_components_match():
    logits, target = make_batch(2, 2, (16, 16, 16), seed=13)
    cc = torch.stack([LabelComponents()(t) for t in target])  # (B, 1, *spatial)
    for fn in (ComponentAdaptiveTverskyLoss(), SmallLesionMILLoss()):
        torch.testing.assert_close(fn(logits, target, components=cc), fn(logits, target))
    d = LabelComponentsd()({"label": target[0].numpy()})
    assert d["label_cc"].shape == target[0].shape


def test_autocast_bf16_no_nan():
    logits, target = make_batch(2, 2, (16, 16, 16), seed=14)
    x = logits.clone().requires_grad_(True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v = CATMILLoss(cat_warmup_epochs=0)(x.to(torch.bfloat16), target)
    v.backward()
    assert v.dtype == torch.float32 and torch.isfinite(v) and torch.isfinite(x.grad).all()


def test_catmil_warmup_and_components():
    logits, target = make_batch(2, 2, (16, 16, 16), seed=15)
    loss = CATMILLoss(lambda_cat=0.1, lambda_mil=0.2, cat_warmup_epochs=50)
    assert loss.set_epoch(0) == 0.0
    assert math.isclose(loss.set_epoch(25), 0.05)
    assert math.isclose(loss.set_epoch(80), 0.1)
    out = loss(logits, target, return_components=True)
    expect = out["base"] + 0.1 * out["cat"] + 0.2 * out["mil"]
    torch.testing.assert_close(out["loss"].detach(), expect)


def test_catmil_warns_without_set_epoch():
    logits, target = make_batch(1, 2, (8, 8, 8), seed=16)
    with pytest.warns(UserWarning, match="set_epoch"):
        CATMILLoss()(logits, target)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CATMILLoss(cat_warmup_epochs=0)(logits, target)


def test_catmil_matches_research_composition():
    """Old trainer: DS(base) + lambda_cat * CAT(out[0]) + lambda_mil * MIL(out[0])."""
    outs, tgts = [], []
    for i, s in enumerate([(32, 32, 32), (16, 16, 16), (8, 8, 8)]):
        o, t = make_batch(2, 2, s, seed=20 + i, p_lesion=0.01)
        outs.append(o)
        tgts.append(t)
    w = nnunet_ds_weights(3)
    base = DiceCELoss(batch_dice=True)
    new = CATMILLoss(base_loss=base, cat_warmup_epochs=0)
    old_cat = ref.ComponentAdaptiveTverskyLoss(w_bg=0.01)
    old_mil = ref.SmallLesionMILLoss()
    old = sum(wi * base(o, t) for wi, o, t in zip(w, outs, tgts) if wi) \
        + 0.1 * old_cat(outs[0], tgts[0]) + 0.2 * old_mil(outs[0], tgts[0])
    torch.testing.assert_close(new(outs, tgts), old.float(), rtol=1e-5, atol=1e-6)
    # ds_mode="all" adds the auxiliary terms at every output
    v_all = CATMILLoss(base_loss=base, cat_warmup_epochs=0, ds_mode="all")(outs, tgts)
    assert torch.isfinite(v_all) and not torch.isclose(v_all, new(outs, tgts))


def test_deep_supervision_wrapper():
    outs = [torch.randn(1, 2, s, s, s) for s in (16, 8, 4)]
    tgts = [(torch.rand(1, 1, s, s, s) > 0.9).float() for s in (16, 8, 4)]
    base = DiceCELoss()
    w = nnunet_ds_weights(3)
    assert np.allclose(w, [2 / 3, 1 / 3, 0])
    expect = w[0] * base(outs[0], tgts[0]) + w[1] * base(outs[1], tgts[1])
    torch.testing.assert_close(DeepSupervisionLoss(base)(outs, tgts), expect.float())
    torch.testing.assert_close(get_loss("dice_ce", deep_supervision=True)(outs, tgts), expect.float())


def test_registry():
    assert {"dice_ce", "tversky", "focal_tversky", "cc_dice_ce", "cat", "mil", "catmil"} <= set(list_losses())
    assert isinstance(get_loss("CC-DiceCE"), torch.nn.Module)
    register_loss("my_loss", lambda: torch.nn.Identity())
    assert "my_loss" in list_losses()
    with pytest.raises(KeyError):
        get_loss("nope")
