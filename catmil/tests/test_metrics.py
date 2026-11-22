import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from catmil.cli import main as cli_main
from catmil.metrics import (lesion_detection_metrics, lesion_size_profile, lesion_table, recall_by_size,
                            remove_small_components)

REPO = Path(__file__).resolve().parents[2]


def _masks(seed, shape=(40, 40, 40)):
    rng = np.random.default_rng(seed)
    gt = ndimage.binary_dilation(rng.random(shape) < 0.002, iterations=int(rng.integers(1, 3)))
    pred = ndimage.binary_dilation(rng.random(shape) < 0.002, iterations=1) | (gt & (rng.random(shape) < 0.5))
    return gt, pred


@pytest.mark.parametrize("seed", range(5))
def test_detection_matches_paper_script(seed):
    if not (REPO / "multiscale_eval.py").exists():
        pytest.skip("paper evaluation script not available")
    sys.path.insert(0, str(REPO))
    try:
        mse = pytest.importorskip("multiscale_eval")
    finally:
        sys.path.pop(0)
    gt, pred = _masks(seed)
    old = mse.lesion_detection_metrics(gt, pred, (1.0, 1.0, 1.2), small_voxels_thresh=50)
    new = lesion_detection_metrics(gt, pred, (1.0, 1.0, 1.2), small_voxels=50)
    for k, v in old.items():
        assert (math.isnan(v) and math.isnan(new[k])) or math.isclose(v, new[k], rel_tol=1e-6, abs_tol=1e-9), k


def test_simple_case():
    gt = np.zeros((10, 10, 10), bool)
    gt[1, 1, 1] = True            # tiny lesion, missed
    gt[5:8, 5:8, 5:8] = True      # 27 voxels, found
    pred = np.zeros_like(gt)
    pred[6, 6, 6] = True
    pred[0, 9, 9] = True          # false positive
    m = lesion_detection_metrics(gt, pred)
    assert m["gt_lesion_count"] == 2 and m["fn_lesion_count"] == 1 and m["fp_lesion_count"] == 1
    assert math.isclose(m["lesion_recall"], 0.5, rel_tol=1e-6)
    r = recall_by_size(gt, pred)
    assert r["tiny"] == {"count": 1, "detected": 0, "recall": 0.0}
    assert r["small"]["count"] == 1 and r["small"]["recall"] == 1.0
    t = lesion_table(gt, pred, prob=gt.astype(float) * 0.7)
    assert list(t["size_vox"]) == [1, 27] and list(t["detected"]) == [False, True]
    assert np.allclose(t["max_prob"], 0.7)
    assert remove_small_components(pred, 2).sum() == 0


def test_profile_cli(tmp_path, capsys):
    for i in range(3):
        gt, _ = _masks(i, (24, 24, 24))
        np.save(tmp_path / f"case{i}.npy", gt.astype(np.uint8))
    assert cli_main(["profile", str(tmp_path), "--json"]) == 0
    out = capsys.readouterr().out
    assert '"lesions"' in out
    prof = lesion_size_profile([np.load(p) for p in sorted(tmp_path.glob("*.npy"))])
    assert prof["cases"] == 3 and prof["lesions"] > 0


def test_bench_cli(capsys):
    assert cli_main(["bench", "--shape", "1", "2", "16", "16", "16", "--iters", "1", "--warmup", "0",
                     "--device", "cpu"]) == 0
    assert "catmil" in capsys.readouterr().out
