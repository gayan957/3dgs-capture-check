"""The command line tool, driven through Typer's CliRunner."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from capturecheck.cli import app, quality_band

runner = CliRunner()


def test_help_lists_the_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("check", "label", "degrade", "train", "evaluate", "export"):
        assert command in result.stdout


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "capturecheck" in result.stdout


def test_labels_command_prints_every_label():
    from capturecheck.labeling import schema

    result = runner.invoke(app, ["labels"])
    assert result.exit_code == 0
    for name in schema.VIDEO_LABEL_NAMES:
        assert name in result.stdout


def test_check_on_a_video(tiny_video):
    result = runner.invoke(app, ["check", str(tiny_video), "--no-dino", "--no-score"])
    assert result.exit_code == 0, result.stdout
    assert "tiny" in result.stdout
    assert "Time:" in result.stdout


def test_check_on_a_photo_folder(tiny_bundle):
    result = runner.invoke(app, ["check", str(tiny_bundle), "--no-dino", "--no-score"])
    assert result.exit_code == 0, result.stdout
    assert "images" in result.stdout


def test_check_on_a_zip(tiny_bundle_zip):
    result = runner.invoke(app, ["check", str(tiny_bundle_zip), "--no-dino", "--no-score"])
    assert result.exit_code == 0, result.stdout


def test_check_json_output(tiny_bundle):
    result = runner.invoke(
        app, ["check", str(tiny_bundle), "--no-dino", "--no-score", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["capture_id"] == "kitchen_photos"
    assert payload["input_type"] == "images"
    assert payload["n_frames"] == 20
    assert set(payload["problems"]) >= {"motion_blur", "poor_coverage"}
    assert payload["score"] is None  # --no-score


def test_label_writes_the_three_csvs(tmp_path, tiny_capture_dir, tiny_bundle):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "kitchen_video").symlink_to(tiny_capture_dir)
    (raw / "kitchen_photos").symlink_to(tiny_bundle)
    out = tmp_path / "labels"

    result = runner.invoke(app, ["label", str(raw), "--out", str(out), "--no-dino"])
    assert result.exit_code == 0, result.stdout
    for name in ("frame_labels.csv", "video_labels.csv", "review_queue.csv"):
        assert (out / name).is_file(), name

    import pandas as pd

    videos = pd.read_csv(out / "video_labels.csv")
    assert set(videos["capture_id"]) == {"kitchen_video", "kitchen_photos"}
    assert videos.loc[videos.capture_id == "kitchen_video", "scene_id"].iloc[0] == "kitchen"

    frames = pd.read_csv(out / "frame_labels.csv")
    assert len(frames) == 24  # 4 video frames + 20 photos


def test_label_on_an_empty_folder_fails_clearly(tmp_path):
    result = runner.invoke(app, ["label", str(tmp_path / "nothing"), "--no-dino"])
    assert result.exit_code == 1
    assert "No captures found" in result.stdout


def test_check_on_a_zip_with_no_photos_fails_clearly(tmp_path):
    """A zip that holds no photos is a user mistake, not a traceback."""
    import zipfile

    empty = tmp_path / "empty.zip"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("notes.txt", "no photos in here")

    result = runner.invoke(app, ["check", str(empty), "--no-dino", "--no-score"])
    assert result.exit_code == 1
    assert "Cannot read" in result.stdout
    assert "no photos" in result.stdout
    assert "Traceback" not in result.stdout


def test_check_on_a_missing_path_fails_clearly(tmp_path):
    result = runner.invoke(app, ["check", str(tmp_path / "nope.mp4"), "--no-dino", "--no-score"])
    assert result.exit_code == 1
    assert "Cannot read" in result.stdout
    assert "Traceback" not in result.stdout


def test_degrade_writes_copies(tmp_path, tiny_bundle):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "kitchen_photos").symlink_to(tiny_bundle)
    out = tmp_path / "degraded"

    result = runner.invoke(
        app, ["degrade", "--raw", str(raw), "--out", str(out), "--strength", "medium"]
    )
    assert result.exit_code == 0, result.stdout
    assert (out / "kitchen_photos__blur_medium" / "labels.json").is_file()


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.95, "likely good"), (0.6, "fair"), (0.4, "likely poor"), (0.1, "likely unusable")],
)
def test_quality_bands(score, expected):
    assert quality_band(score) == expected


def test_check_without_a_trained_model_says_so(tiny_bundle, monkeypatch):
    """No run yet must read as "not trained", not as "you asked me not to"."""
    from capturecheck.model import predict as predict_mod

    monkeypatch.setattr(predict_mod, "load_predictor", lambda *a, **k: None)
    result = runner.invoke(app, ["check", str(tiny_bundle), "--no-dino"])
    assert result.exit_code == 0, result.stdout
    assert "no trained model yet" in result.stdout
    assert "capturecheck train" in result.stdout


def test_check_reports_a_score_when_a_run_exists(tiny_bundle, monkeypatch):
    from capturecheck.model import predict as predict_mod
    from capturecheck.model.predict import Prediction

    monkeypatch.setattr(
        predict_mod,
        "predict_capture",
        lambda result, run_dir=None: Prediction(
            capture_id=result.capture_id,
            score=0.73,
            problems={"motion_blur": 0.9},
            run_id="2026-01-01_run01",
        ),
    )
    result = runner.invoke(app, ["check", str(tiny_bundle), "--no-dino"])
    assert result.exit_code == 0, result.stdout
    assert "0.73" in result.stdout
    assert "2026-01-01_run01" in result.stdout


def test_json_output_is_strictly_valid(tiny_bundle):
    """`--json` is for piping, so a missing measurement must be null, not NaN."""
    from capturecheck.cli import json_safe

    result = runner.invoke(
        app, ["check", str(tiny_bundle), "--no-dino", "--no-score", "--json"]
    )
    assert result.exit_code == 0, result.stdout
    assert "NaN" not in result.stdout
    assert "Infinity" not in result.stdout
    # json.loads accepts NaN, so parse strictly instead.
    json.loads(result.stdout, parse_constant=_reject)

    assert json_safe(float("nan")) is None
    assert json_safe(float("inf")) is None
    assert json_safe({"a": [float("nan"), 1.5]}) == {"a": [None, 1.5]}
    assert json_safe(np_float()) == 2.5


def _reject(name):
    raise AssertionError(f"{name} is not valid JSON")


def np_float():
    import numpy as np

    return np.float32(2.5)
