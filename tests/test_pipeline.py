"""The end-to-end labelling pipeline on the synthetic fixtures.

DINOv2 is switched off (``use_dino=False``) so the tests stay offline and fast;
the DINOv2 path has its own tests in ``test_similarity.py``.
"""

from __future__ import annotations

import json

import pytest

from capturecheck.io.load import load_capture
from capturecheck.labeling import schema
from capturecheck.labeling.pipeline import label_capture, measure_capture


def test_label_capture_on_a_video(tiny_video):
    result = label_capture(tiny_video, use_dino=False)
    assert result.capture_id == "tiny"
    assert len(result.frame_rows) == 4
    assert set(schema.FRAME_LABEL_NAMES).issubset(result.frame_rows.columns)

    row = result.video_row
    assert row["input_type"] == "video"
    assert row["n_frames"] == 4
    for name in schema.VIDEO_LABEL_NAMES:
        assert row[name] in (0, 1)
    assert isinstance(json.loads(row["evidence"]), dict)
    assert row["label_source"] == "auto"
    assert row["time_s_total"] < 60


def test_label_capture_on_a_photo_bundle(tiny_bundle):
    result = label_capture(tiny_bundle, scene_id="kitchen", use_dino=False)
    row = result.video_row
    assert row["input_type"] == "images"
    assert row["scene_id"] == "kitchen"
    assert row["n_frames"] == 20
    assert row["n_cameras"] == 1
    assert result.frame_rows["frame_name"].iloc[0] == "IMG_0000.jpg"
    # EXIF made it into the frame table.
    assert result.frame_rows["exposure_s"].iloc[0] == 0.008
    assert result.frame_rows["iso"].iloc[0] == 200


def test_zip_and_folder_give_the_same_labels(tiny_bundle, tiny_bundle_zip):
    folder = label_capture(tiny_bundle, use_dino=False).video_row
    zipped = label_capture(tiny_bundle_zip, use_dino=False).video_row
    for name in schema.VIDEO_LABEL_NAMES:
        assert folder[name] == zipped[name], name


def test_degrade_labels_override_the_rules(tiny_bundle):
    result = label_capture(
        tiny_bundle,
        use_dino=False,
        degrade_labels={"motion_blur": 1},
        degrade_type="blur",
        degrade_strength="heavy",
    )
    assert result.video_row["motion_blur"] == 1
    assert result.video_row["label_source"] == "degrade_script"
    assert result.video_row["degrade_type"] == "blur"


def test_measure_capture_fills_every_measurement(tiny_bundle):
    capture = load_capture(tiny_bundle)
    measurements = measure_capture(capture, use_dino=False)
    frame_df = measurements.frame_df
    for column in ("sharpness", "brightness", "noise", "features", "overlap_next"):
        assert column in frame_df.columns
    assert frame_df["sharpness"].notna().all()
    # The last frame has no successor, so its overlap is empty by design.
    assert frame_df["overlap_next"].iloc[:-1].notna().all()
    assert not measurements.pair_df.empty
    assert measurements.timings.total > 0
    assert "quality" in measurements.timings.steps


def test_video_pairs_get_a_flow_speed(tiny_video):
    capture = load_capture(tiny_video)
    measurements = measure_capture(capture, use_dino=False)
    assert measurements.pair_df["flow_speed"].notna().any()


def test_photo_pairs_have_no_flow_speed(tiny_bundle):
    capture = load_capture(tiny_bundle)
    measurements = measure_capture(capture, use_dino=False)
    assert measurements.pair_df["flow_speed"].isna().all()


def test_cached_measurements_round_trip(tmp_path, tiny_bundle, monkeypatch):
    """Training reads pairs and coverage from the cache, not from the CSVs.

    Without this the model trains with no pair tokens and an empty coverage
    grid, then is asked to score captures that have both.
    """
    from capturecheck.config import get_config
    from capturecheck.labeling.pipeline import load_measurements

    cfg = get_config()
    monkeypatch.setattr(cfg.paths, "features", str(tmp_path.relative_to(tmp_path.anchor)))
    monkeypatch.setattr(type(cfg.paths), "resolve", lambda self, key: tmp_path)

    result = label_capture(
        tiny_bundle, use_dino=False, cache_measurements=True, keep_measurements=True
    )
    pair_df, grid = load_measurements(result.capture_id)

    assert not pair_df.empty
    assert len(pair_df) == len(result.measurements.pair_df)
    assert "walked" in pair_df.columns
    assert set(pair_df["walked"].dropna().unique()) <= {0.0, 1.0}
    assert grid is not None and grid.shape == (8, 3)


def test_training_and_scoring_see_the_same_shapes(tmp_path, tiny_bundle, monkeypatch):
    import pandas as pd

    from capturecheck.config import get_config
    from capturecheck.labeling.pipeline import load_measurements
    from capturecheck.model.data import build_sample
    from capturecheck.model.predict import sample_from_result

    cfg = get_config()
    monkeypatch.setattr(type(cfg.paths), "resolve", lambda self, key: tmp_path)

    result = label_capture(
        tiny_bundle, use_dino=False, cache_measurements=True, keep_measurements=True
    )
    live = sample_from_result(result)

    # What the training path rebuilds from the CSVs plus the cache.
    pair_df, grid = load_measurements(result.capture_id)
    rebuilt = build_sample(
        capture_id=result.capture_id,
        scene_id=result.capture_id,
        frame_df=result.frame_rows,
        pair_df=pair_df,
        video_row=result.video_row,
        emb=result.measurements.emb,
        emb_idx=result.measurements.emb_idx,
        coverage_grid=grid,
    )

    assert rebuilt.pairs.shape == live.pairs.shape
    assert rebuilt.frames.shape == live.frames.shape
    assert rebuilt.global_vec.shape == live.global_vec.shape
    # The coverage grid must carry the same information either way.
    import numpy as np

    assert np.allclose(rebuilt.global_vec[-24:], live.global_vec[-24:], equal_nan=True)
    assert isinstance(pair_df, pd.DataFrame)


def test_video_and_bundle_of_the_same_scene_measure_alike(tmp_path, bundle_frames):
    """The same content as a video and as photos must not land in different worlds."""
    from conftest import write_photo, write_video

    folder = tmp_path / "as_photos"
    folder.mkdir()
    for i, frame in enumerate(bundle_frames):
        write_photo(folder / f"{i:04d}.jpg", frame, when=None)

    # 2 fps sampling of a 30 fps video recovers one frame per held pose.
    held = [f for frame in bundle_frames for f in [frame] * 15]
    video = write_video(tmp_path / "as_video.mp4", held, fps=30.0)

    photos = label_capture(folder, use_dino=False)
    movie = label_capture(video, use_dino=False)

    assert photos.video_row["input_type"] == "images"
    assert movie.video_row["input_type"] == "video"
    assert abs(photos.video_row["n_frames"] - movie.video_row["n_frames"]) <= 1

    # The frame-level measurements should agree closely; JPEG vs mp4 explains the rest.
    assert movie.video_row["blur_ratio"] == pytest.approx(
        photos.video_row["blur_ratio"], abs=0.25
    )
    # Only the video gets a flow speed, and only photos carry EXIF columns.
    assert movie.frame_rows["time_s"].notna().all()
    assert photos.frame_rows["time_s"].isna().all()
