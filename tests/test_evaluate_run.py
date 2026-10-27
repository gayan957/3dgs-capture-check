"""The whole evaluation, end to end: metrics, baselines.json and the report."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from helpers import make_sample

from capturecheck.labeling import schema
from capturecheck.model import evaluate as evaluate_mod
from capturecheck.model.data import FeatureStats
from capturecheck.model.net import CaptureQualityNet


@pytest.fixture()
def run_dir(tmp_path):
    """A saved two-seed run, as train_run writes one."""
    folder = tmp_path / "2026-01-01_run01"
    folder.mkdir()
    settings = {"d_model": 32, "layers": 1, "heads": 2, "dropout": 0.0}
    states = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        states.append(CaptureQualityNet(**settings).state_dict())
    torch.save({"states": states, "seeds": [0, 1], "model": settings}, folder / "model.pt")
    FeatureStats.fit([make_sample(seed=i) for i in range(6)]).save(
        folder / "feature_stats.json"
    )
    (folder / "metrics.json").write_text(json.dumps({"run_id": folder.name, "n_seeds": 2}))
    (folder / "splits.json").write_text(
        json.dumps({"train_scenes": ["a", "b"], "test_scenes": ["c"]})
    )
    return folder


def _test_set() -> list:
    """One scene's worth of captures: a clean one and copies of rising damage."""
    samples = []
    for index, (capture_id, strength, damage) in enumerate(
        [
            ("c_real", "", 0),
            ("c__blur_light", "light", 1),
            ("c__blur_medium", "medium", 2),
            ("c__blur_heavy", "heavy", 3),
            ("c__noise_light", "light", 1),
            ("c__noise_medium", "medium", 2),
            ("c__noise_heavy", "heavy", 3),
            ("c__subsample_heavy", "heavy", 3),
        ]
    ):
        sample = make_sample(
            capture_id=capture_id,
            scene_id="c",
            source="real" if damage == 0 else "degraded",
            degrade_type="none" if damage == 0 else capture_id.split("__")[1].rsplit("_", 1)[0],
            degrade_strength=strength,
            input_type="images",
            seed=100 + index,
        )
        sample.problems[schema.PROBLEM_LABELS.index("motion_blur")] = float(
            "blur" in capture_id
        )
        samples.append(sample)
    return samples


@pytest.fixture()
def wired(monkeypatch, tmp_path):
    train = [make_sample(capture_id=f"t{i}", scene_id="a", psnr=20.0 + i, seed=i) for i in range(8)]
    test = _test_set()
    monkeypatch.setattr(
        evaluate_mod, "samples_from_labels", lambda **kwargs: train + test
    )
    monkeypatch.setattr(evaluate_mod, "read_splits", lambda: (["a", "b"], ["c"]))
    # The runtime figure comes from the label CSVs; there are none here.
    import pandas as pd

    from capturecheck.io import dataset

    monkeypatch.setattr(dataset, "read_video_labels", lambda *a, **k: pd.DataFrame())
    return train, test


def test_evaluate_run_writes_everything(run_dir, wired, tmp_path, monkeypatch):
    from capturecheck.config import get_config

    cfg = get_config()
    reports = tmp_path / "reports"
    monkeypatch.setattr(
        type(cfg.paths), "resolve",
        lambda self, key: reports if key == "reports" else tmp_path,
    )

    report_path = tmp_path / "evaluation_report.md"
    result = evaluate_mod.evaluate_run(run_dir=run_dir, report_path=report_path)

    assert result.n_test == 8
    assert {m.name for m in result.methods} == {
        "capturecheck", "frame_count", "mean_blur", "orb_matches", "lightgbm_features"
    }
    assert result.model_score is not None

    # metrics.json gains a test section without losing what training wrote.
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["n_seeds"] == 2
    assert metrics["test"]["n_captures"] == 8
    assert "spearman" in metrics["test"]
    assert len(metrics["test"]["problems"]) == schema.N_PROBLEMS

    baselines = json.loads((run_dir / "baselines.json").read_text())
    assert "capturecheck" not in baselines
    assert set(baselines) == {
        "frame_count", "mean_blur", "orb_matches", "lightgbm_features"
    }


def test_the_report_reads_as_a_report(run_dir, wired, tmp_path, monkeypatch):
    from capturecheck.config import get_config

    cfg = get_config()
    monkeypatch.setattr(type(cfg.paths), "resolve", lambda self, key: tmp_path)
    report_path = tmp_path / "evaluation_report.md"
    result = evaluate_mod.evaluate_run(run_dir=run_dir, report_path=report_path)

    text = report_path.read_text()
    assert text.startswith("# Evaluation report")
    assert "2026-01-01_run01" in text
    assert "## Ranking quality" in text
    assert "## Problems found" in text
    assert "## Runtime" in text
    assert "## Honest notes" in text
    for name in schema.PROBLEM_LABELS:
        assert f"`{name}`" in text
    # Every method appears in the ranking table.
    for method in result.methods:
        assert method.name in text


def test_no_psnr_falls_back_to_damage_order_and_says_so(run_dir, wired, tmp_path, monkeypatch):
    from capturecheck.config import get_config

    cfg = get_config()
    monkeypatch.setattr(type(cfg.paths), "resolve", lambda self, key: tmp_path)
    result = evaluate_mod.evaluate_run(
        run_dir=run_dir, report_path=tmp_path / "report.md"
    )
    assert result.n_with_psnr == 0
    assert any("damage severity" in note for note in result.notes)
    assert "damage severity" in (tmp_path / "report.md").read_text()


def test_a_tiny_test_set_is_refused(run_dir, monkeypatch, tmp_path):
    monkeypatch.setattr(
        evaluate_mod, "samples_from_labels",
        lambda **kwargs: [make_sample(capture_id="only", scene_id="c")],
    )
    monkeypatch.setattr(evaluate_mod, "read_splits", lambda: (["a"], ["c"]))
    with pytest.raises(RuntimeError, match="need at least 3"):
        evaluate_mod.evaluate_run(run_dir=run_dir, report_path=tmp_path / "r.md")


def test_evaluating_without_a_run_fails_clearly(tmp_path, monkeypatch):
    from capturecheck.model import train as train_mod

    monkeypatch.setattr(train_mod, "latest_run", lambda models_dir=None: None)
    with pytest.raises(RuntimeError, match="no trained run"):
        evaluate_mod.evaluate_run()


def test_a_scatter_plot_is_written(run_dir, wired, tmp_path, monkeypatch):
    from capturecheck.config import get_config

    cfg = get_config()
    reports = tmp_path / "reports"
    monkeypatch.setattr(type(cfg.paths), "resolve", lambda self, key: reports)
    evaluate_mod.evaluate_run(run_dir=run_dir, report_path=tmp_path / "r.md")
    figures = list((reports / "figures").glob("*.png"))
    assert len(figures) == 1
    assert figures[0].stat().st_size > 1000


def test_an_equal_baseline_is_called_out(run_dir, wired, tmp_path, monkeypatch):
    """If something simple ranks as well, the report has to say so."""
    from capturecheck.config import get_config

    cfg = get_config()
    monkeypatch.setattr(type(cfg.paths), "resolve", lambda self, key: tmp_path)

    # Force a baseline to match the model exactly.
    real_baselines = evaluate_mod.baseline_mod.all_baselines

    def rigged(train, test, colmap_db=None, seed=0):
        results = real_baselines(train, test, colmap_db, seed)
        results[0].scores = np.arange(float(len(test)))
        return results

    monkeypatch.setattr(evaluate_mod.baseline_mod, "all_baselines", rigged)
    monkeypatch.setattr(
        evaluate_mod, "spearman", lambda a, b: 0.8  # every method ties
    )
    result = evaluate_mod.evaluate_run(run_dir=run_dir, report_path=tmp_path / "r.md")
    assert any("not yet earning its place" in note for note in result.notes)
