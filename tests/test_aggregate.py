"""Frame labels -> capture labels, evidence, and the review decision."""

from __future__ import annotations

import numpy as np
import pandas as pd

from capturecheck.checks.coverage import CameraPath, Coverage, coverage_map
from capturecheck.config import get_config
from capturecheck.labeling.aggregate import (
    describe_range,
    longest_run,
    summarize_labels,
    to_video_labels,
)
from capturecheck.labeling.confidence import needs_review
from capturecheck.labeling.frame_rules import label_frames


def make_frames(n: int = 40, is_video: bool = True, **overrides) -> pd.DataFrame:
    """A table of ``n`` good frames, with named columns overridden wholesale."""
    data = {
        "capture_id": ["cap"] * n,
        "frame_idx": list(range(n)),
        "frame_name": [f"frame_{i:03d}" if is_video else f"IMG_{i:04d}.jpg" for i in range(n)],
        "time_s": [i * 0.5 for i in range(n)] if is_video else [None] * n,
        "sharpness": [500.0] * n,
        "brightness": [120.0] * n,
        "clip_dark": [0.0] * n,
        "clip_bright": [0.0] * n,
        "noise": [1.0] * n,
        "features": [800] * n,
        "dup_of_prev": [0.5] * n,
        "shiny_prob": [0.1] * n,
        "overlap_next": [0.6] * (n - 1) + [np.nan],
    }
    data.update(overrides)
    return label_frames(pd.DataFrame(data))


def make_pairs(n: int = 20, kind: str = "walked") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "i": list(range(n)),
            "j": [i + 2 for i in range(n)],
            "kind": [kind] * n,
            "confidence": [0.8] * n,
            "flow_speed": [4.0] * n,
            "brightness_delta": [1.0] * n,
        }
    )


def good_coverage() -> Coverage:
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    centers = np.stack([np.sin(angles), np.zeros(24), np.cos(angles)], axis=1) * 5.0
    rotations = []
    for a in angles:
        c, s = np.cos(a + np.pi), np.sin(a + np.pi)
        rotations.append(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
    return coverage_map(CameraPath(centers=centers, rotations=np.array(rotations)))


def test_clean_capture_has_no_problems():
    result = to_video_labels(
        make_frames(), make_pairs(), good_coverage(),
        info={"input_type": "video"}, diversity=0.4, unique_views=40,
    )
    assert all(v == 0 for v in result.labels.values()), result.labels
    assert result.evidence == {}
    assert result.stats["blur_ratio"] == 0.0
    assert result.stats["walked_ratio"] == 1.0


def test_motion_blur_is_flagged_with_a_time_range():
    n = 40
    sharp = [500.0] * n
    for i in range(10, 25):  # a run of blurry frames
        sharp[i] = 20.0
    result = to_video_labels(
        make_frames(n, sharpness=sharp), make_pairs(), good_coverage(),
        info={"input_type": "video"}, diversity=0.4, unique_views=40,
    )
    assert result.labels["motion_blur"] == 1
    assert "5.0-12.0 s (frames 10-24)" in result.evidence["motion_blur"]
    assert "38% of frames blurry" in result.evidence["motion_blur"]


def test_photo_bundle_evidence_uses_file_names():
    n = 30
    sharp = [500.0] * n
    for i in range(5, 20):
        sharp[i] = 20.0
    frames = make_frames(n, is_video=False, sharpness=sharp)
    result = to_video_labels(
        frames, make_pairs(), good_coverage(),
        info={"input_type": "images", "n_cameras": 1, "n_sizes": 1},
        diversity=0.4, unique_views=30,
    )
    assert result.labels["motion_blur"] == 1
    assert "IMG_0005.jpg to IMG_0019.jpg" in result.evidence["motion_blur"]


def test_exif_shutter_speed_is_added_to_blur_evidence():
    n = 30
    frames = make_frames(
        n, is_video=False, sharpness=[20.0] * n, exposure_s=[1 / 30] * n, iso=[100] * n
    )
    result = to_video_labels(
        frames, make_pairs(), good_coverage(),
        info={"input_type": "images"}, diversity=0.4, unique_views=30,
    )
    assert "shutter 1/30 s" in result.evidence["motion_blur"]


def test_no_walking_when_pairs_only_turned():
    result = to_video_labels(
        make_frames(), make_pairs(kind="turned"), good_coverage(),
        info={"input_type": "video"}, diversity=0.4, unique_views=40,
    )
    assert result.labels["no_walking"] == 1
    assert "rotation on the spot" in result.evidence["no_walking"]


def test_poor_coverage_needs_a_reliable_path():
    empty = Coverage(
        grid=np.zeros((8, 3), dtype=np.int32), empty_bins=24, empty_bins_all=24,
        empty_names=["left, low"], reliable=False,
    )
    walked = to_video_labels(
        make_frames(), make_pairs(), empty,
        info={"input_type": "video"}, diversity=0.4, unique_views=40,
    )
    # The path failed but the camera did move: do not claim poor coverage.
    assert walked.labels["poor_coverage"] == 0

    pivoted = to_video_labels(
        make_frames(), make_pairs(kind="turned"), empty,
        info={"input_type": "video"}, diversity=0.4, unique_views=40,
    )
    assert pivoted.labels["no_walking"] == 1
    assert pivoted.labels["poor_coverage"] == 1


def test_low_overlap_points_at_the_gap():
    n = 20
    overlaps = [0.6] * (n - 1) + [np.nan]
    for i in range(3, 10):
        overlaps[i] = 0.05
    frames = make_frames(n, is_video=False, overlap_next=overlaps)
    result = to_video_labels(
        frames, make_pairs(), good_coverage(),
        info={"input_type": "images"}, diversity=0.4, unique_views=20,
    )
    assert result.labels["low_overlap"] == 1
    assert "gap between IMG_0003.jpg and IMG_0004.jpg" in result.evidence["low_overlap"]


def test_mixed_cameras_only_for_bundles():
    info = {"input_type": "images", "n_cameras": 2, "n_sizes": 2, "cameras": ["A", "B"]}
    bundle = to_video_labels(
        make_frames(is_video=False), make_pairs(), good_coverage(),
        info=info, diversity=0.4, unique_views=40,
    )
    assert bundle.labels["mixed_cameras"] == 1
    assert "2 cameras" in bundle.evidence["mixed_cameras"]

    video = to_video_labels(
        make_frames(), make_pairs(), good_coverage(),
        info={**info, "input_type": "video"}, diversity=0.4, unique_views=40,
    )
    assert video.labels["mixed_cameras"] == 0


def test_too_few_views_and_reflective_and_texture():
    cfg = get_config().labeling
    n = 40
    result = to_video_labels(
        make_frames(n, shiny_prob=[0.9] * n, features=[10] * n),
        make_pairs(), good_coverage(),
        info={"input_type": "video"}, diversity=0.01, unique_views=3,
    )
    assert result.labels["too_few_views"] == 1
    assert result.labels["reflective"] == 1
    assert result.labels["low_texture"] == 1
    assert result.stats["shiny_ratio"] > cfg.shiny.video_ratio


def test_longest_run_and_describe_range():
    mask = np.array([False, True, True, False, True, True, True, False])
    assert longest_run(mask) == (4, 6)
    assert longest_run(np.zeros(5, dtype=bool)) is None

    video = make_frames(10)
    assert describe_range(video, 2, 4, True) == "1.0-2.0 s (frames 2-4)"
    photos = make_frames(10, is_video=False)
    assert describe_range(photos, 2, 4, False) == "IMG_0002.jpg to IMG_0004.jpg"
    assert describe_range(photos, 3, 3, False) == "IMG_0003.jpg"


def test_summarize_labels():
    rows = pd.DataFrame([{"motion_blur": 1, "lighting": 0}, {"motion_blur": 1, "lighting": 0}])
    table = summarize_labels(rows)
    blur = table[table["label"] == "motion_blur"].iloc[0]
    assert blur["captures"] == 2 and blur["share"] == 1.0


def test_needs_review_on_a_borderline_value():
    cfg = get_config().labeling
    row = {
        "blur_ratio": cfg.blur.video_blur_ratio,  # exactly on the threshold
        "n_frames": 300,
        "overlap_mean": 0.5,
        "coverage_empty_bins": 0,
    }
    decision = needs_review(row, cfg)
    assert decision.needs_review
    assert "blur_ratio" in decision.reason


def test_needs_review_when_checks_disagree():
    cfg = get_config().labeling
    row = {
        "diversity": 0.6,
        "coverage_empty_bins": cfg.coverage.max_empty_bins + 10,
        "n_frames": 300,
        "overlap_mean": 0.5,
    }
    decision = needs_review(row, cfg)
    assert decision.needs_review
    assert "camera path may be wrong" in decision.reason


def test_clean_row_needs_no_review():
    cfg = get_config().labeling
    row = {
        "blur_ratio": 0.0,
        "brightness_std": 2.0,
        "walked_ratio": 0.95,
        "overlap_low_ratio": 0.0,
        "shiny_ratio": 0.0,
        "low_texture_ratio": 0.0,
        "noise_ratio": 0.0,
        "unique_views": 60,
        "coverage_empty_bins": 1,
        "diversity": 0.4,
        "overlap_mean": 0.6,
        "flow_speed_mean": 3.0,
        "n_frames": 300,
    }
    flag, reason = needs_review(row, cfg)
    assert flag is False and reason == ""
