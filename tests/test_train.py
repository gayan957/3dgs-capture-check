"""A real (tiny) training run, end to end.

Catches the things unit tests on single functions miss: that the two stages wire
together, that a run folder ends up holding everything a predictor needs, and
that the ensemble can then score a capture.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml
from helpers import make_sample

from capturecheck.config import load_config
from capturecheck.model import train as train_mod
from capturecheck.model.data import FeatureStats
from capturecheck.model.predict import RunPredictor


def tiny_dataset() -> list:
    """Three scenes, each a clean capture plus light/medium/heavy broken copies."""
    samples = []
    for scene_index, scene in enumerate(["kitchen", "office", "studio"]):
        samples.append(
            make_sample(
                capture_id=f"{scene}_real",
                scene_id=scene,
                psnr=24.0 + scene_index,
                seed=scene_index * 10,
            )
        )
        for strength_index, strength in enumerate(["light", "medium", "heavy"]):
            samples.append(
                make_sample(
                    capture_id=f"{scene}__blur_{strength}",
                    scene_id=scene,
                    source="degraded",
                    degrade_type="blur",
                    degrade_strength=strength,
                    seed=scene_index * 10 + strength_index + 1,
                )
            )
    return samples


@pytest.fixture()
def fast_config(tmp_path, monkeypatch):
    """A config with the stages cut down to a few steps."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "train.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": [0, 1],
                "model": {"d_model": 32, "layers": 1, "heads": 2, "dropout": 0.1,
                          "max_tokens": 80},
                "stage_a": {"epochs": 1, "variants_per_epoch": 8, "batch_size": 4},
                "stage_b": {"epochs": 2, "loss": "ranking", "aux_pretext_weight": 0.3,
                            "problem_weight": 1.0, "batch_size": 4, "margin": 0.1,
                            "patience": 2},
                "optimizer": {"lr": 3.0e-4, "weight_decay": 1.0e-2},
                "eval": {"split": "leave_one_scene_out", "bootstrap": 50},
            }
        )
    )
    monkeypatch.setenv("CAPTURECHECK_CONFIG_DIR", str(config_dir))
    from capturecheck import config as config_mod

    cached = config_mod.get_config
    cached.cache_clear()
    cfg = load_config(config_dir)
    monkeypatch.setattr(config_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(train_mod, "get_config", lambda: cfg)
    yield cfg
    cached.cache_clear()


@pytest.fixture()
def wired(monkeypatch, fast_config):
    samples = tiny_dataset()
    monkeypatch.setattr(train_mod, "samples_from_labels", lambda **kwargs: samples)
    monkeypatch.setattr(
        train_mod, "read_splits", lambda: (["kitchen", "office"], ["studio"])
    )
    return samples


def test_train_run_writes_a_complete_run(tmp_path, wired):
    result = train_mod.train_run(models_dir=tmp_path, run_id="test_run")
    run_dir = result.run_dir
    assert run_dir == tmp_path / "test_run"

    for name in (
        "model.pt", "feature_stats.json", "config.yaml",
        "thresholds.yaml", "splits.json", "metrics.json", "train_log.csv",
    ):
        assert (run_dir / name).is_file(), name

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_seeds"] == 2
    assert metrics["n_train_captures"] == 8  # two scenes, four captures each
    assert metrics["n_test_captures"] == 4

    splits = json.loads((run_dir / "splits.json").read_text())
    assert splits["test_scenes"] == ["studio"]
    assert set(splits["val_scene_per_seed"].values()) <= {"kitchen", "office"}


def test_both_stages_run_and_are_logged(tmp_path, wired):
    import pandas as pd

    train_mod.train_run(models_dir=tmp_path, run_id="staged")
    log = pd.read_csv(tmp_path / "staged" / "train_log.csv")
    assert set(log["stage"]) == {"A", "B"}
    assert log["seed"].nunique() == 2


def test_stage_a_can_be_skipped(tmp_path, wired):
    import pandas as pd

    train_mod.train_run(models_dir=tmp_path, run_id="b_only", run_stage_a=False)
    log = pd.read_csv(tmp_path / "b_only" / "train_log.csv")
    assert set(log["stage"]) == {"B"}


def test_the_trained_run_can_score_a_capture(tmp_path, wired):
    result = train_mod.train_run(models_dir=tmp_path, run_id="scoring")
    predictor = RunPredictor(result.run_dir, prefer_onnx=False)
    assert len(predictor.models) == 2

    prediction = predictor.predict(make_sample(capture_id="unseen", seed=99))
    assert 0.0 <= prediction.score <= 1.0
    assert len(prediction.score_per_seed) == 2
    assert set(prediction.problems) and all(
        0.0 <= p <= 1.0 for p in prediction.problems.values()
    )
    assert prediction.top_frames  # attention pointed at some frames


def test_feature_stats_come_from_the_train_side_only(tmp_path, wired, monkeypatch):
    """Test scenes must not influence the normalisation."""
    captured = {}
    original = FeatureStats.fit

    def spy(samples):
        captured["scenes"] = {s.scene_id for s in samples}
        return original(samples)

    monkeypatch.setattr(train_mod.FeatureStats, "fit", staticmethod(spy))
    train_mod.train_run(models_dir=tmp_path, run_id="stats")
    assert captured["scenes"] == {"kitchen", "office"}
    assert "studio" not in captured["scenes"]


def test_each_seed_holds_out_a_different_scene(tmp_path, wired):
    result = train_mod.train_run(models_dir=tmp_path, run_id="loso")
    val_scenes = [seed.val_scene for seed in result.seeds]
    assert val_scenes == ["kitchen", "office"]


def test_training_fails_clearly_without_splits(tmp_path, monkeypatch, fast_config):
    monkeypatch.setattr(train_mod, "read_splits", lambda: ([], []))
    with pytest.raises(RuntimeError, match="no train scenes"):
        train_mod.train_run(models_dir=tmp_path)


def test_latest_run_picks_the_newest(tmp_path, wired):
    import time

    train_mod.train_run(models_dir=tmp_path, run_id="older")
    time.sleep(0.01)
    train_mod.train_run(models_dir=tmp_path, run_id="newer")
    assert train_mod.latest_run(models_dir=tmp_path).name == "newer"


def test_ranking_accuracy_is_a_share():
    scores = np.array([0.9, 0.1, 0.5])
    import torch

    better = torch.tensor([0, 2])
    worse = torch.tensor([1, 1])
    value = train_mod.ranking_accuracy(torch.from_numpy(scores), better, worse)
    assert value == pytest.approx(1.0)
