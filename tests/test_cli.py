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
