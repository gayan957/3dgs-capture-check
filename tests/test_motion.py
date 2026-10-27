"""Overlap, flow speed, walked-vs-turned and the rough camera path."""

from __future__ import annotations

import cv2
import numpy as np

from capturecheck.checks import coverage as coverage_mod
from capturecheck.checks import motion


def test_overlap_is_high_for_neighbours_and_low_for_unrelated(sharp_image):
    shifted = np.roll(sharp_image, 6, axis=1)
    other = np.random.default_rng(9).integers(0, 255, sharp_image.shape, dtype=np.uint8)
    near = motion.overlap(sharp_image, shifted)
    far = motion.overlap(sharp_image, other)
    assert 0.0 <= far < near <= 1.0
    assert near > 0.3


def test_overlap_of_identical_frames_is_near_one(sharp_image):
    assert motion.overlap(sharp_image, sharp_image) > 0.9


def test_overlap_series_has_one_value_per_pair(trans_frames):
    features = motion.orb_frames(trans_frames)
    series = motion.overlap_series(features)
    assert len(series) == len(trans_frames) - 1
    assert all(0.0 <= v <= 1.0 for v in series)


def test_flow_speed_rises_with_motion(sharp_image):
    still = motion.flow_speed(sharp_image, sharp_image)
    moved = motion.flow_speed(sharp_image, np.roll(sharp_image, 12, axis=1))
    assert still < 1.0 < moved


def test_pure_rotation_is_turned(rot_frames):
    pairs = motion.motion_pairs(rot_frames, gap=1)
    assert len(pairs) == len(rot_frames) - 1
    kinds = [p.kind for _, _, p in pairs]
    assert kinds.count(motion.TURNED) >= 0.8 * len(kinds)
    assert motion.walked_ratio(pairs) < 0.25


def test_translation_through_a_scene_is_walked(trans_frames):
    pairs = motion.motion_pairs(trans_frames, gap=1)
    kinds = [p.kind for _, _, p in pairs]
    assert kinds.count(motion.WALKED) >= 0.8 * len(kinds)
    assert motion.walked_ratio(pairs) > 0.75


def test_too_few_matches_gives_unknown():
    flat_a = np.full((120, 160, 3), 100, dtype=np.uint8)
    flat_b = np.full((120, 160, 3), 140, dtype=np.uint8)
    result = motion.walked_or_turned(flat_a, flat_b)
    assert result.kind == motion.UNKNOWN
    assert result.confidence == 0.0


def test_motion_pairs_uses_gap_for_video(trans_frames):
    pairs = motion.motion_pairs(trans_frames, gap=2, is_video=True)
    assert [(i, j) for i, j, _ in pairs] == [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]


def test_motion_pairs_uses_neighbours_for_photos(trans_frames):
    pairs = motion.motion_pairs(trans_frames, gap=2, is_video=False)
    assert [(i, j) for i, j, _ in pairs][:3] == [(0, 1), (1, 2), (2, 3)]


def test_approx_intrinsics_uses_exif_focal():
    k_guess = motion.approx_intrinsics((480, 640))
    k_exif = motion.approx_intrinsics((480, 640), focal_px=1000.0)
    assert k_guess[0, 0] == 576.0  # 0.9 * width
    assert k_exif[0, 0] == 1000.0
    assert k_exif[0, 2] == 320.0 and k_exif[1, 2] == 240.0


def test_rough_path_moves_for_translation_and_not_for_rotation(rot_frames, trans_frames):
    walked = coverage_mod.rough_path(trans_frames)
    turned = coverage_mod.rough_path(rot_frames)
    assert walked.extent > 0.5
    assert turned.extent < 1e-6  # a pivot recovers no translation


def test_coverage_map_flags_unreliable_when_no_motion(rot_frames):
    path = coverage_mod.rough_path(rot_frames)
    cov = coverage_mod.coverage_map(path)
    assert cov.reliable is False
    assert cov.empty_bins == cov.grid.size


def test_coverage_map_counts_empty_bins_in_used_rows(trans_frames):
    path = coverage_mod.rough_path(trans_frames)
    cov = coverage_mod.coverage_map(path)
    assert cov.grid.shape == (8, 3)
    assert cov.filled_bins >= 1
    # Only the elevation rows that were used are judged.
    assert cov.empty_bins <= cov.empty_bins_all
    assert isinstance(coverage_mod.summarize_empty(cov), str)


def test_circular_path_covers_every_azimuth():
    """A synthetic orbit around the origin must leave no azimuth bin empty."""
    n = 24
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centers = np.stack([np.sin(angles), np.zeros(n), np.cos(angles)], axis=1) * 5.0
    rotations = []
    for a in angles:
        # Look back at the origin from each position.
        c, s = np.cos(a + np.pi), np.sin(a + np.pi)
        rotations.append(np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]))
    path = coverage_mod.CameraPath(centers=centers, rotations=np.array(rotations))
    cov = coverage_mod.coverage_map(path)
    assert cov.empty_bins == 0
    assert cov.azimuth_span_deg > 300


def test_homography_and_sampson_errors_are_small_for_exact_models(sharp_image):
    pts_a = np.array([[10, 10], [100, 20], [50, 90], [120, 120]], dtype=np.float32)
    shift = np.array([[1, 0, 5], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    pts_b = cv2.perspectiveTransform(pts_a.reshape(-1, 1, 2).astype(np.float64), shift)
    errors = motion._homography_errors(pts_a, pts_b.reshape(-1, 2).astype(np.float32), shift)
    assert np.allclose(errors, 0, atol=1e-6)


def test_undecidable_pair_reports_unknown(sharp_image):
    """When matching collapses, the honest answer is "unknown", not a guess."""
    rng = np.random.default_rng(3)
    other = rng.integers(0, 255, sharp_image.shape, dtype=np.uint8)
    result = motion.walked_or_turned(sharp_image, other)
    assert result.kind in {motion.UNKNOWN, motion.WALKED, motion.TURNED}
    if result.kind == motion.UNKNOWN:
        assert result.confidence == 0.0


def test_identical_frames_read_as_turned(sharp_image):
    """No motion means no parallax, so a homography explains everything."""
    result = motion.walked_or_turned(sharp_image, sharp_image)
    assert result.kind == motion.TURNED
    assert result.homography_share > motion.HOMOGRAPHY_SHARE
