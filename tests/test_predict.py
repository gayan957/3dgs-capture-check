"""Scoring a capture with a saved run."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from helpers import make_sample

from capturecheck.labeling import schema
from capturecheck.model.data import FeatureStats
from capturecheck.model.net import CaptureQualityNet
from capturecheck.model.predict import RunPredictor, load_predictor, sample_from_result


@pytest.fixture()
def saved_run(tmp_path):
    """A two-member ensemble written to disk the way train_run writes one."""
    run_dir = tmp_path / "2026-01-01_run01"
    run_dir.mkdir()
    settings = {"d_model": 32, "layers": 1, "heads": 2, "dropout": 0.0}
    states = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        states.append(CaptureQualityNet(**settings).state_dict())
    torch.save({"states": states, "seeds": [0, 1], "model": settings}, run_dir / "model.pt")
    FeatureStats.fit([make_sample(seed=i) for i in range(4)]).save(
        run_dir / "feature_stats.json"
    )
    return run_dir


def test_predictor_averages_the_ensemble(saved_run):
    predictor = RunPredictor(saved_run, prefer_onnx=False)
    prediction = predictor.predict(make_sample(capture_id="x", seed=7))

    assert prediction.backend == "torch"
    assert prediction.run_id == "2026-01-01_run01"
    assert len(prediction.score_per_seed) == 2
    assert prediction.score == pytest.approx(float(np.mean(prediction.score_per_seed)))
    assert 0.0 <= prediction.score <= 1.0


def test_every_problem_gets_a_probability(saved_run):
    prediction = RunPredictor(saved_run, prefer_onnx=False).predict(make_sample())
    assert set(prediction.problems) == set(schema.PROBLEM_LABELS)
    assert all(0.0 <= p <= 1.0 for p in prediction.problems.values())


def test_attention_points_at_frames(saved_run):
    sample = make_sample(n_frames=20, n_pairs=8)
    prediction = RunPredictor(saved_run, prefer_onnx=False).predict(sample)
    assert prediction.attention is not None
    assert prediction.attention.shape[0] == 1 + 20 + 8
    assert len(prediction.top_frames) == 5
    assert all(0 <= i < 20 for i in prediction.top_frames)


def test_prediction_serialises(saved_run):
    payload = RunPredictor(saved_run, prefer_onnx=False).predict(make_sample()).as_dict()
    assert set(payload) >= {"capture_id", "score", "problems", "run_id", "backend"}
    assert isinstance(payload["score"], float)


def test_load_predictor_returns_none_without_a_run(tmp_path, monkeypatch):
    from capturecheck.model import train as train_mod

    monkeypatch.setattr(train_mod, "latest_run", lambda models_dir=None: None)
    assert load_predictor(run_dir=tmp_path / "nothing") is None


def test_sample_from_result_uses_the_measurements(tiny_bundle):
    """Scoring must not re-measure a capture the pipeline already measured."""
    from capturecheck.labeling.pipeline import label_capture
    from capturecheck.model.data import FRAME_DIM, GLOBAL_DIM

    result = label_capture(tiny_bundle, use_dino=False, keep_measurements=True)
    sample = sample_from_result(result)
    assert sample.capture_id == "kitchen_photos"
    assert sample.frames.shape[1] == FRAME_DIM
    assert sample.global_vec.shape == (GLOBAL_DIM,)
    assert sample.pairs.shape[0] > 0  # pair measurements carried over


def test_sample_from_result_without_measurements(tiny_bundle):
    from capturecheck.labeling.pipeline import label_capture

    result = label_capture(tiny_bundle, use_dino=False, keep_measurements=False)
    sample = sample_from_result(result)
    assert sample.frames.shape[0] > 0
    assert sample.pairs.shape[0] == 0  # no pair table was kept
