"""Building model samples from the label CSVs.

A blank CSV cell arrives as NaN, which is truthy. That detail silently filed
every clean capture under degrade type "nan", left stage A with nothing to
damage, and skipped the pretext stage entirely. These tests pin it down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from capturecheck.labeling import schema
from capturecheck.model.data import (
    CaptureDataset,
    _number,
    _text,
    build_sample,
    clean_samples,
    samples_from_labels,
)


def test_text_treats_blanks_and_nan_as_absent():
    assert _text(float("nan"), "none") == "none"
    assert _text(None, "none") == "none"
    assert _text("", "none") == "none"
    assert _text("  ", "none") == "none"
    assert _text("nan", "none") == "none"  # a CSV round trip can stringify it
    assert _text("blur", "none") == "blur"
    assert _text(" blur ", "none") == "blur"


def test_number_treats_blanks_and_nan_as_the_default():
    assert _number(float("nan")) == 0.0
    assert _number(None) == 0.0
    assert _number("") == 0.0
    assert _number("1") == 1.0
    assert _number(2.5) == 2.5


def _video_row(**overrides) -> dict:
    row = {name: 0 for name in schema.VIDEO_LABEL_NAMES}
    row.update(
        capture_id="cap", scene_id="scene", input_type="images", n_frames=10,
        source="real", degrade_type=None, degrade_strength=None, psnr=None,
    )
    row.update(overrides)
    return row


def _frame_rows(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "capture_id": ["cap"] * n,
            "frame_idx": range(n),
            "frame_name": [f"{i:04d}.jpg" for i in range(n)],
            "sharpness": [500.0] * n,
            "brightness": [120.0] * n,
            "features": [800] * n,
            "noise": [1.0] * n,
            "overlap_next": [0.5] * n,
        }
    )


def test_a_clean_capture_read_from_csv_is_recognised_as_clean():
    """The bug: a blank degrade_type became the string "nan"."""
    row = _video_row(degrade_type=float("nan"), degrade_strength=float("nan"))
    sample = build_sample(
        capture_id="cap", scene_id="scene", frame_df=_frame_rows(),
        pair_df=pd.DataFrame(), video_row=row,
        emb=np.zeros((0, 384), dtype=np.float32), emb_idx=np.zeros(0, dtype=int),
    )
    assert sample.degrade_type == "none"
    assert sample.degrade_strength == ""
    assert sample.damage_rank == 0.0
    assert clean_samples([sample]) == [sample]


def test_a_degraded_capture_keeps_its_damage_rank():
    row = _video_row(source="degraded", degrade_type="blur", degrade_strength="heavy")
    sample = build_sample(
        capture_id="cap", scene_id="scene", frame_df=_frame_rows(),
        pair_df=pd.DataFrame(), video_row=row,
        emb=np.zeros((0, 384), dtype=np.float32), emb_idx=np.zeros(0, dtype=int),
    )
    assert sample.degrade_type == "blur"
    assert sample.damage_rank == 3.0
    assert clean_samples([sample]) == []


def test_problem_labels_survive_a_blank_cell():
    row = _video_row(motion_blur=1, lighting=float("nan"))
    sample = build_sample(
        capture_id="cap", scene_id="scene", frame_df=_frame_rows(),
        pair_df=pd.DataFrame(), video_row=row,
        emb=np.zeros((0, 384), dtype=np.float32), emb_idx=np.zeros(0, dtype=int),
    )
    assert sample.problems[schema.PROBLEM_LABELS.index("motion_blur")] == 1.0
    assert sample.problems[schema.PROBLEM_LABELS.index("lighting")] == 0.0


def test_samples_round_trip_through_the_csvs(tmp_path):
    """Write the CSVs, read them back, and the clean captures must still be clean."""
    from capturecheck.io import dataset

    rows = [
        _video_row(capture_id="kitchen_real", scene_id="kitchen"),
        _video_row(
            capture_id="kitchen__blur_light", scene_id="kitchen",
            source="degraded", degrade_type="blur", degrade_strength="light",
        ),
    ]
    dataset.write_video_labels(pd.DataFrame(rows), out_dir=tmp_path)
    frames = pd.concat(
        [_frame_rows().assign(capture_id=row["capture_id"]) for row in rows], ignore_index=True
    )
    dataset.write_frame_labels(frames, out_dir=tmp_path)

    samples = samples_from_labels(labels_dir=tmp_path, features_dir=tmp_path / "features")
    assert len(samples) == 2
    by_id = {s.capture_id: s for s in samples}
    assert by_id["kitchen_real"].degrade_type == "none"
    assert by_id["kitchen_real"].damage_rank == 0.0
    assert by_id["kitchen__blur_light"].damage_rank == 1.0
    assert len(clean_samples(samples)) == 1


# --------------------------------------------------------------------------- #
# CaptureDataset
# --------------------------------------------------------------------------- #
def _dataset() -> CaptureDataset:
    from helpers import make_sample

    return CaptureDataset(
        [
            make_sample(capture_id="a_real", scene_id="a", psnr=25.0),
            make_sample(
                capture_id="a_blur", scene_id="a", source="degraded",
                degrade_type="blur", degrade_strength="medium", seed=1,
            ),
            make_sample(capture_id="b_real", scene_id="b", input_type="video", seed=2),
        ]
    )


def test_dataset_is_a_collection():
    data = _dataset()
    assert len(data) == 3
    assert data[0].capture_id == "a_real"
    assert [s.capture_id for s in data] == ["a_real", "a_blur", "b_real"]
    assert data.scenes == ["a", "b"]


def test_dataset_subsets():
    data = _dataset()
    assert [s.capture_id for s in data.for_scenes({"a"})] == ["a_real", "a_blur"]
    assert [s.capture_id for s in data.excluding_scene("a")] == ["b_real"]
    assert [s.capture_id for s in data.clean()] == ["a_real", "b_real"]
    assert [s.capture_id for s in data.with_psnr()] == ["a_real"]


def test_dataset_stats_and_summary():
    data = _dataset()
    assert data.problem_matrix().shape == (3, schema.N_PROBLEMS)
    stats = data.fit_stats()
    assert np.isfinite(stats.frame_mean).all()
    assert "3 captures" in data.summary()
    assert "1 video" in data.summary()


def test_empty_dataset_is_harmless():
    data = CaptureDataset([])
    assert len(data) == 0
    assert data.scenes == []
    assert data.problem_matrix().shape == (0, schema.N_PROBLEMS)
    with pytest.raises(IndexError):
        _ = data[0]
