"""nnU-Net compatibility: base loss parity and the trainers' loss construction (needs nnunetv2)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

nnunetv2 = pytest.importorskip("nnunetv2")
from nnunetv2.training.loss.compound_losses import DC_and_BCE_loss, DC_and_CE_loss  # noqa: E402
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper  # noqa: E402
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss  # noqa: E402

import _reference_losses as ref  # noqa: E402
from catmil import CCDiceCEGlobalLoss, DiceCELoss  # noqa: E402
from catmil.integrations import nnunet as integ  # noqa: E402
from conftest import make_batch  # noqa: E402

TOL = dict(rtol=1e-5, atol=1e-6)


def _nn_dice_ce(batch_dice, smooth=1e-5, ignore_label=None):
    return DC_and_CE_loss({'batch_dice': batch_dice, 'smooth': smooth, 'do_bg': False, 'ddp': False}, {},
                          weight_ce=1, weight_dice=1, ignore_label=ignore_label,
                          dice_class=MemoryEfficientSoftDiceLoss)


@pytest.mark.parametrize("batch_dice", [False, True])
@pytest.mark.parametrize("c", [2, 4])
def test_dice_ce_matches_nnunet(batch_dice, c):
    logits, target = make_batch(2, c, (16, 16, 16), seed=30, p_lesion=0.01)
    torch.testing.assert_close(DiceCELoss(batch_dice=batch_dice)(logits, target),
                               _nn_dice_ce(batch_dice)(logits, target), **TOL)


def test_dice_ce_ignore_label_matches_nnunet():
    logits, target = make_batch(2, 2, (16, 16, 16), seed=31)
    target[:, :, :5] = 2
    torch.testing.assert_close(DiceCELoss(batch_dice=True, ignore_index=2)(logits, target),
                               _nn_dice_ce(True, ignore_label=2)(logits, target), **TOL)


@pytest.mark.parametrize("use_ignore", [False, True])
def test_dice_bce_regions_matches_nnunet(use_ignore):
    logits = torch.randn(2, 3, 16, 16, 16)
    tgt = (torch.rand(2, 3, 16, 16, 16) > 0.9).float()
    if use_ignore:
        tgt = torch.cat([tgt, (torch.rand(2, 1, 16, 16, 16) > 0.7).float()], 1)
    nn_loss = DC_and_BCE_loss({}, {'batch_dice': True, 'do_bg': True, 'smooth': 1e-5, 'ddp': False},
                              use_ignore_label=use_ignore, dice_class=MemoryEfficientSoftDiceLoss)
    ours = DiceCELoss(batch_dice=True, activation="sigmoid", ignore_index=1 if use_ignore else None)
    torch.testing.assert_close(ours(logits, tgt), nn_loss(logits, tgt), **TOL)


def test_cc_dice_ce_global_matches_research():
    logits, target = make_batch(2, 2, (16, 16, 16), seed=32, p_lesion=0.01)
    kw = dict(spacing=(1, 1, 2), patch_size=(16, 16, 16))
    expect = ref.CCDiceCELoss(**kw)(logits, target) + _nn_dice_ce(True, smooth=0)(logits, target)
    torch.testing.assert_close(CCDiceCEGlobalLoss(batch_dice=True, **kw)(logits, target), expect, **TOL)


# ---------------------------------------------------------------- trainers (no dataset needed)

def _fake_trainer(cls, has_regions=False, ignore_label=None, heads=2, ds=True, epoch=0):
    """Instance with only the attributes _build_loss uses (nnU-Net's __init__ needs a dataset)."""
    t = cls.__new__(cls)
    t.label_manager = SimpleNamespace(has_regions=has_regions, ignore_label=ignore_label,
                                      has_ignore_label=ignore_label is not None, num_segmentation_heads=heads)
    t.configuration_manager = SimpleNamespace(batch_dice=True, spacing=[1.0, 1.0, 2.0], patch_size=[32, 32, 32])
    t.enable_deep_supervision = ds
    t.is_ddp = False
    t.device = torch.device("cpu")
    t.current_epoch = epoch
    t._get_deep_supervision_scales = lambda: [[1, 1, 1], [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]]
    t.print_to_log_file = lambda *a, **k: None
    return t


def _ds_batch(c=2):
    outs, tgts = [], []
    for i, s in enumerate([(32, 32, 32), (16, 16, 16), (8, 8, 8)]):
        o, t = make_batch(2, c, s, seed=40 + i, p_lesion=0.01)
        outs.append(o)
        tgts.append(t)
    return outs, tgts


W = np.array([1.0, 0.5, 0.0]) / 1.5  # nnU-Net weights for 3 outputs


@pytest.mark.parametrize("epoch,lam_cat", [(0, 0.0), (25, 0.05), (60, 0.1)])
def test_catmil_trainer_matches_research_trainer(epoch, lam_cat):
    t = _fake_trainer(integ.nnUNetTrainerCATMIL, epoch=epoch)
    loss = t._build_loss()
    outs, tgts = _ds_batch()
    base = DeepSupervisionWrapper(_nn_dice_ce(True), W)
    expect = base(outs, tgts) + lam_cat * ref.ComponentAdaptiveTverskyLoss(w_bg=0.01)(outs[0], tgts[0]) \
        + 0.2 * ref.SmallLesionMILLoss()(outs[0], tgts[0])
    torch.testing.assert_close(loss(outs, tgts), expect.float(), **TOL)


def test_cat_and_mil_trainers():
    outs, tgts = _ds_batch()
    base = DeepSupervisionWrapper(_nn_dice_ce(True), W)(outs, tgts)
    cat = _fake_trainer(integ.nnUNetTrainerCAT, epoch=100)._build_loss()
    torch.testing.assert_close(cat(outs, tgts), (base + 0.1 * ref.ComponentAdaptiveTverskyLoss(w_bg=0.01)(
        outs[0], tgts[0])).float(), **TOL)
    mil = _fake_trainer(integ.nnUNetTrainerMIL)._build_loss()
    torch.testing.assert_close(mil(outs, tgts), (base + 0.2 * ref.SmallLesionMILLoss()(outs[0], tgts[0])).float(),
                               **TOL)


def test_baseline_trainers_match_research():
    outs, tgts = _ds_batch()
    tv = _fake_trainer(integ.nnUNetTrainerTversky)._build_loss()
    torch.testing.assert_close(tv(outs, tgts), DeepSupervisionWrapper(ref.MyTverskyLoss(0.3, 0.7), W)(outs, tgts),
                               **TOL)
    ft = _fake_trainer(integ.nnUNetTrainerFocalTversky)._build_loss()
    torch.testing.assert_close(ft(outs, tgts), DeepSupervisionWrapper(ref.FocalTverskyLoss(0.3, 0.7, 4 / 3), W)(
        outs, tgts), **TOL)
    cc = _fake_trainer(integ.nnUNetTrainerCCDiceCE)._build_loss()
    sp, ps = [1.0, 1.0, 2.0], [32, 32, 32]
    expect = DeepSupervisionWrapper(ref._Sum(ref.CCDiceCELoss(sp, ps), _nn_dice_ce(True, smooth=0)), W)(outs, tgts)
    torch.testing.assert_close(cc(outs, tgts), expect, **TOL)


def test_trainer_regions_ignore_and_no_ds():
    # region-based with ignore channel
    t = _fake_trainer(integ.nnUNetTrainerCATMIL, has_regions=True, ignore_label=3, epoch=60)
    loss = t._build_loss()
    outs = [torch.randn(2, 2, s, s, s) for s in (16, 8, 4)]
    tgts = [torch.cat([(torch.rand(2, 2, s, s, s) > 0.95).float(), (torch.rand(2, 1, s, s, s) > 0.8).float()], 1)
            for s in (16, 8, 4)]
    assert torch.isfinite(loss(outs, tgts))
    # deep supervision disabled -> plain tensors
    t = _fake_trainer(integ.nnUNetTrainerCATMIL, ds=False, epoch=60)
    o, y = make_batch(2, 2, (16, 16, 16), seed=50)
    assert torch.isfinite(t._build_loss()(o, y))


def test_hyperparameters_from_subclass_and_env(monkeypatch):
    class Variant(integ.nnUNetTrainerCATMIL):
        catmil_lambda_mil = 0.5

    assert _fake_trainer(Variant)._build_loss().lambda_mil == 0.5
    monkeypatch.setenv("CATMIL_LAMBDA_MIL", "0.7")
    monkeypatch.setenv("CATMIL_DS_MODE", "all")
    loss = _fake_trainer(integ.nnUNetTrainerCATMIL)._build_loss()
    assert loss.lambda_mil == 0.7 and loss.ds_mode == "all"


def test_trainers_do_not_override_init():
    """nnU-Net stores the base __init__ arguments; subclasses must not change the signature."""
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    for name in integ.__all__:
        assert getattr(integ, name).__init__ is nnUNetTrainer.__init__, name
