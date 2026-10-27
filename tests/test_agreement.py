"""The label-agreement report (scripts/07).

The reviewer rows here are synthetic, made inside the test: this exercises the
maths and the report, it does not stand in for a real review of real captures.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capturecheck.io import dataset
from capturecheck.labeling import schema


@pytest.fixture(scope="module")
def agreement():
    path = Path(__file__).resolve().parents[1] / "scripts" / "07_check_label_agreement.py"
    spec = importlib.util.spec_from_file_location("agreement_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["agreement_script"] = module
    spec.loader.exec_module(module)
    return module


def test_cohen_kappa_bounds(agreement):
    same = np.array([1, 0, 1, 0, 1, 0])
    assert agreement.cohen_kappa(same, same) == pytest.approx(1.0)
    assert agreement.cohen_kappa(same, 1 - same) == pytest.approx(-1.0)
    # Chance agreement scores zero, not the raw 50%.
    a = np.array([1, 1, 0, 0])
    b = np.array([1, 0, 1, 0])
    assert agreement.cohen_kappa(a, b) == pytest.approx(0.0)


def test_cohen_kappa_when_everyone_says_no(agreement):
    zeros = np.zeros(6, dtype=int)
    assert agreement.cohen_kappa(zeros, zeros) == 1.0


def test_precision_recall(agreement):
    truth = np.array([1, 1, 1, 0])
    predicted = np.array([1, 1, 0, 1])
    precision, recall, f1 = agreement.precision_recall(truth, predicted)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(2 / 3)


def _human_rows(capture_id: str, reviewer: str, **labels) -> dict:
    row = {name: 0 for name in schema.VIDEO_LABEL_NAMES}
    row.update(labels)
    row.update(
        capture_id=capture_id,
        scene_id=capture_id,
        input_type="images",
        note="",
        reviewer=reviewer,
        timestamp="2026-01-01T00:00:00",
        label_source="human",
    )
    return row


def test_agreed_labels_take_the_majority(agreement):
    human = pd.DataFrame(
        [
            _human_rows("a", "ann", motion_blur=1),
            _human_rows("a", "bob", motion_blur=0),  # a disagreement
            _human_rows("b", "ann", motion_blur=1),
            _human_rows("b", "bob", motion_blur=1),
        ]
    )
    agreed, pairs = agreement.agreed_labels(human)
    by_id = agreed.set_index("capture_id")
    assert by_id.loc["b", "motion_blur"] == 1
    assert by_id.loc["a", "motion_blur__disagreed"] == 1
    assert by_id.loc["b", "motion_blur__disagreed"] == 0
    assert pairs["capture_id"].nunique() == 2


def test_reviewer_kappa_uses_the_first_two_reviewers(agreement):
    human = pd.DataFrame(
        [
            _human_rows("a", "ann", lighting=1),
            _human_rows("a", "bob", lighting=1),
            _human_rows("b", "ann", lighting=0),
            _human_rows("b", "bob", lighting=0),
        ]
    )
    _, pairs = agreement.agreed_labels(human)
    kappa, n = agreement.reviewer_kappa(pairs, "lighting")
    assert n == 2
    assert kappa == pytest.approx(1.0)


def test_report_is_written_with_precision_per_label(agreement, tmp_path, monkeypatch):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    auto_rows, human_rows = [], []
    for i in range(12):
        capture_id = f"cap{i:02d}"
        truth = 1 if i < 6 else 0
        # The automatic label is right except for two false alarms.
        auto = truth if i not in (6, 7) else 1
        row = {name: 0 for name in schema.VIDEO_LABEL_NAMES}
        row.update(
            capture_id=capture_id, scene_id=capture_id, input_type="images",
            motion_blur=auto, lighting=truth, source="real", label_source="auto",
        )
        auto_rows.append(row)
        for reviewer in ("ann", "bob"):
            human_rows.append(
                _human_rows(capture_id, reviewer, motion_blur=truth, lighting=truth)
            )

    dataset.write_video_labels(pd.DataFrame(auto_rows), out_dir=labels_dir)
    dataset.append_human_labels(pd.DataFrame(human_rows), out_dir=labels_dir)

    report = tmp_path / "label_agreement.md"
    monkeypatch.setattr(
        sys, "argv",
        ["07", "--labels", str(labels_dir), "--out", str(report), "--min-precision", "0.8"],
    )
    agreement.main()

    text = report.read_text()
    assert "# Label agreement" in text
    assert "`motion_blur`" in text and "`lighting`" in text
    # 6 true positives, 2 false alarms -> precision 0.75, recall 1.00
    assert "0.75" in text and "1.00" in text
    # Below the 0.8 bar, so it must be listed with the threshold to change.
    assert "blur.sharp_min_laplacian_var" in text
    assert "## Where reviewers disagreed" not in text  # the reviewers agreed


def test_missing_human_labels_is_a_clear_error(agreement, tmp_path, monkeypatch):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    row = {name: 0 for name in schema.VIDEO_LABEL_NAMES}
    row.update(capture_id="a", scene_id="a", input_type="images")
    dataset.write_video_labels(pd.DataFrame([row]), out_dir=labels_dir)

    monkeypatch.setattr(
        sys, "argv", ["07", "--labels", str(labels_dir), "--out", str(tmp_path / "r.md")]
    )
    with pytest.raises(SystemExit, match="no human labels"):
        agreement.main()
