"""Adding a capture must cost one capture, and must never mix two rule sets.

The label CSVs are the training table, so a row appended to them has to have
been measured exactly the way the rows already there were measured. These tests
pin down both halves: the cheap path when the settings are unchanged, and the
refusal when they are not.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from capturecheck.cli import app
from capturecheck.io import dataset
from capturecheck.labeling.fingerprint import labeling_digest

runner = CliRunner()


@pytest.fixture
def raw_with_two(tmp_path, tiny_capture_dir, tiny_bundle) -> Path:
    """A raw folder holding one video capture and one photo bundle."""
    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copytree(tiny_capture_dir, raw / "first_video")
    shutil.copytree(tiny_bundle, raw / "first_bundle")
    return raw


def _label(raw: Path, out: Path, *extra: str):
    return runner.invoke(app, ["label", str(raw), "--out", str(out), "--no-dino", *extra])


def _ids(out: Path) -> set[str]:
    return set(pd.read_csv(out / "video_labels.csv")["capture_id"].astype(str))


def test_the_first_run_measures_everything_and_records_how(raw_with_two, tmp_path):
    out = tmp_path / "labels"
    result = _label(raw_with_two, out)
    assert result.exit_code == 0, result.output
    assert "no labels here yet" in result.output
    assert _ids(out) == {"first_video", "first_bundle"}

    run = json.loads((out / "label_run.json").read_text())
    assert run["config_digest"] == labeling_digest(use_dino=False)
    assert run["n_captures"] == 2


def test_a_new_capture_is_detected_and_only_it_is_measured(raw_with_two, tmp_path, tiny_bundle):
    out = tmp_path / "labels"
    _label(raw_with_two, out)
    shutil.copytree(tiny_bundle, raw_with_two / "second_bundle")

    result = _label(raw_with_two, out)
    assert result.exit_code == 0, result.output
    # Only the new capture was measured, and the other two were carried over.
    assert "measured: 1 new" in result.output
    assert "kept: 2" in result.output
    assert _ids(out) == {"first_video", "first_bundle", "second_bundle"}


def test_the_carried_rows_are_the_same_rows_not_remeasured(raw_with_two, tmp_path, tiny_bundle):
    out = tmp_path / "labels"
    _label(raw_with_two, out)
    before = pd.read_csv(out / "video_labels.csv").set_index("capture_id")

    shutil.copytree(tiny_bundle, raw_with_two / "second_bundle")
    _label(raw_with_two, out)
    after = pd.read_csv(out / "video_labels.csv").set_index("capture_id")

    for capture_id in ("first_video", "first_bundle"):
        pd.testing.assert_series_equal(
            before.loc[capture_id], after.loc[capture_id], check_names=False
        )


def test_frame_rows_follow_the_capture(raw_with_two, tmp_path, tiny_bundle):
    out = tmp_path / "labels"
    _label(raw_with_two, out)
    shutil.copytree(tiny_bundle, raw_with_two / "second_bundle")
    _label(raw_with_two, out)

    frames = pd.read_csv(out / "frame_labels.csv")
    assert set(frames["capture_id"].astype(str)) == {
        "first_video",
        "first_bundle",
        "second_bundle",
    }


def test_nothing_new_is_a_no_op_that_says_so(raw_with_two, tmp_path):
    out = tmp_path / "labels"
    _label(raw_with_two, out)
    stamp = (out / "video_labels.csv").stat().st_mtime_ns

    result = _label(raw_with_two, out)
    assert result.exit_code == 0, result.output
    assert "Already up to date" in result.output
    # The table was not rewritten at all.
    assert (out / "video_labels.csv").stat().st_mtime_ns == stamp


def test_a_removed_capture_loses_its_rows(raw_with_two, tmp_path):
    out = tmp_path / "labels"
    _label(raw_with_two, out)
    shutil.rmtree(raw_with_two / "first_bundle")

    result = _label(raw_with_two, out)
    assert result.exit_code == 0, result.output
    assert "removed: 1" in result.output
    assert _ids(out) == {"first_video"}
    frames = pd.read_csv(out / "frame_labels.csv")
    assert "first_bundle" not in set(frames["capture_id"].astype(str))


def test_a_changed_threshold_forces_a_full_remeasure(raw_with_two, tmp_path, monkeypatch):
    out = tmp_path / "labels"
    _label(raw_with_two, out)

    # Move a threshold: every row on disk was measured against the old one.
    from capturecheck.config import get_config

    cfg = get_config()
    monkeypatch.setattr(cfg.labeling.overlap, "min_overlap", 0.9)

    result = _label(raw_with_two, out)
    assert result.exit_code == 0, result.output
    assert "the labelling settings changed" in result.output
    assert "measured: 1 new" not in result.output
    run = json.loads((out / "label_run.json").read_text())
    assert run["config_digest"] == labeling_digest(use_dino=False)


def test_dropping_dino_changes_the_digest(raw_with_two, tmp_path):
    """Without DINOv2 a capture has no diversity, duplicate or shiny
    measurement at all, so those rows cannot share a table with rows that do.
    Checked on the digest rather than by running the real embedder, so this
    stays offline."""
    assert labeling_digest(use_dino=True) != labeling_digest(use_dino=False)

    out = tmp_path / "labels"
    _label(raw_with_two, out)
    recorded = json.loads((out / "label_run.json").read_text())["config_digest"]
    assert recorded == labeling_digest(use_dino=False)
    assert recorded != labeling_digest(use_dino=True)


def test_new_only_refuses_rather_than_remeasure(raw_with_two, tmp_path, monkeypatch):
    out = tmp_path / "labels"
    _label(raw_with_two, out)
    before = (out / "video_labels.csv").read_text()

    from capturecheck.config import get_config

    monkeypatch.setattr(get_config().labeling.overlap, "min_overlap", 0.9)
    result = _label(raw_with_two, out, "--new-only")

    assert result.exit_code == 1
    assert "Cannot label only the new captures" in result.output
    assert "--from-scratch" in result.output
    # It refused, so it changed nothing.
    assert (out / "video_labels.csv").read_text() == before


def test_new_only_on_a_fresh_folder_refuses(raw_with_two, tmp_path):
    out = tmp_path / "labels"
    result = _label(raw_with_two, out, "--new-only")
    assert result.exit_code == 1
    assert "no labels here yet" in result.output
    assert not (out / "video_labels.csv").exists()


def test_from_scratch_remeasures_everything(raw_with_two, tmp_path):
    out = tmp_path / "labels"
    _label(raw_with_two, out)
    result = _label(raw_with_two, out, "--from-scratch")

    assert result.exit_code == 0, result.output
    assert "--from-scratch was given" in result.output
    assert "kept:" not in result.output
    assert _ids(out) == {"first_video", "first_bundle"}


def test_the_two_flags_together_are_refused(raw_with_two, tmp_path):
    out = tmp_path / "labels"
    result = _label(raw_with_two, out, "--from-scratch", "--new-only")
    assert result.exit_code == 2
    assert "cannot both be given" in result.output


def test_an_unreadable_run_file_forces_a_full_remeasure(raw_with_two, tmp_path):
    out = tmp_path / "labels"
    _label(raw_with_two, out)
    (out / "label_run.json").write_text("{not json")

    result = _label(raw_with_two, out)
    assert result.exit_code == 0, result.output
    assert "the labelling settings changed" in result.output


def test_incremental_and_full_runs_agree(raw_with_two, tmp_path, tiny_bundle):
    """The whole point: the cheap path must produce the same table."""
    shutil.copytree(tiny_bundle, raw_with_two / "second_bundle")

    full = tmp_path / "full"
    _label(raw_with_two, full)

    step = tmp_path / "step"
    moved = tmp_path / "held"
    shutil.move(raw_with_two / "second_bundle", moved)
    _label(raw_with_two, step)
    shutil.move(moved, raw_with_two / "second_bundle")
    _label(raw_with_two, step)

    def table(folder: Path) -> pd.DataFrame:
        df = pd.read_csv(folder / "video_labels.csv").sort_values("capture_id")
        # ``time_s_total`` is how long measuring took, not a measurement.
        df = df.drop(columns=["time_s_total"]).reset_index(drop=True)
        return df[sorted(df.columns)]

    pd.testing.assert_frame_equal(table(full), table(step))


def test_the_digest_is_stable_across_processes():
    """A digest from ``hash()`` would differ every run and break every append."""
    import subprocess
    import sys

    code = (
        "from capturecheck.labeling.fingerprint import labeling_digest;"
        "print(labeling_digest(use_dino=True))"
    )
    seen = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(seen) == 1
    assert seen.pop() == labeling_digest(use_dino=True)


def test_review_queue_is_rebuilt_from_the_whole_table(raw_with_two, tmp_path, tiny_bundle):
    out = tmp_path / "labels"
    _label(raw_with_two, out)
    shutil.copytree(tiny_bundle, raw_with_two / "second_bundle")
    _label(raw_with_two, out)

    videos = pd.read_csv(out / "video_labels.csv")
    queue = dataset.read_review_queue(out_dir=out)
    assert len(queue) == int((videos["review"] == 1).sum())
    assert set(queue["capture_id"].astype(str)) <= set(videos["capture_id"].astype(str))
