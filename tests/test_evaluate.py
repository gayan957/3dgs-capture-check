"""Evaluation maths and the baselines."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import make_sample

from capturecheck.model import baselines as baseline_mod
from capturecheck.model.data import GLOBAL_FEATURES
from capturecheck.model.evaluate import bootstrap_ci, precision_recall_f1, spearman


def test_spearman_is_rank_based():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a * 10) == pytest.approx(1.0)
    assert spearman(a, -a) == pytest.approx(-1.0)
    # A monotone but non-linear relation is still a perfect rank correlation.
    assert spearman(a, a**3) == pytest.approx(1.0)


def test_spearman_ignores_missing_values():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, b) == pytest.approx(1.0)


def test_spearman_needs_enough_points():
    assert np.isnan(spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])))


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    a = rng.normal(size=60)
    b = a + rng.normal(scale=0.4, size=60)
    point = spearman(a, b)
    lo, hi = bootstrap_ci(a, b, n_boot=300, seed=0)
    assert lo < point < hi
    assert -1.0 <= lo < hi <= 1.0


def test_bootstrap_is_undefined_for_a_tiny_sample():
    lo, hi = bootstrap_ci(np.arange(3.0), np.arange(3.0), n_boot=100)
    assert np.isnan(lo) and np.isnan(hi)


def test_precision_recall_f1():
    truth = np.array([1, 1, 0, 0])
    predicted = np.array([1, 0, 1, 0])
    precision, recall, f1 = precision_recall_f1(truth, predicted)
    assert precision == pytest.approx(0.5)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(0.5)


def test_precision_is_undefined_when_nothing_was_predicted():
    precision, recall, _ = precision_recall_f1(np.array([1, 0]), np.array([0, 0]))
    assert np.isnan(precision)
    assert recall == 0.0


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def _with_global(name: str, value: float, **kwargs):
    sample = make_sample(**kwargs)
    sample.global_vec[GLOBAL_FEATURES.index(name)] = value
    return sample


def test_frame_count_baseline_reads_the_frame_count():
    samples = [
        _with_global("n_frames", 10.0, capture_id="a"),
        _with_global("n_frames", 400.0, capture_id="b"),
    ]
    result = baseline_mod.frame_count_baseline(samples)
    assert result.scores.tolist() == [10.0, 400.0]


def test_mean_blur_baseline_reads_the_sharpness():
    samples = [
        _with_global("sharpness_mean", 50.0, capture_id="a"),
        _with_global("sharpness_mean", 900.0, capture_id="b"),
    ]
    assert baseline_mod.mean_blur_baseline(samples).scores.tolist() == [50.0, 900.0]


def test_match_count_falls_back_to_orb_and_says_so():
    samples = [make_sample(capture_id="a"), make_sample(capture_id="b")]
    result = baseline_mod.match_count_baseline(samples, colmap_db=None)
    assert result.name == "orb_matches"
    assert "not a COLMAP match count" in result.note


def test_lightgbm_baseline_learns_the_ordering():
    """Trained on PSNR, it should put a high-PSNR capture above a low one."""
    train = []
    for i in range(24):
        psnr = 20.0 + i * 0.4
        sample = _with_global("n_frames", float(20 + 8 * i), capture_id=f"t{i}", psnr=psnr, seed=i)
        train.append(sample)
    test = [
        _with_global("n_frames", 30.0, capture_id="low", seed=100),
        _with_global("n_frames", 210.0, capture_id="high", seed=101),
    ]
    result = baseline_mod.lightgbm_baseline(train, test, seed=0)
    assert result.scores[1] > result.scores[0]


def test_all_baselines_returns_one_score_per_capture():
    train = [make_sample(capture_id=f"t{i}", psnr=20.0 + i, seed=i) for i in range(8)]
    test = [make_sample(capture_id=f"e{i}", seed=100 + i) for i in range(4)]
    results = baseline_mod.all_baselines(train, test)
    assert {r.name for r in results} == {
        "frame_count", "mean_blur", "orb_matches", "lightgbm_features"
    }
    for result in results:
        assert len(result.scores) == len(test)
