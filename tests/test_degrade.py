"""Broken copies: each op must change the right measurement and label it exactly."""

from __future__ import annotations

import json

import numpy as np

from capturecheck.checks import quality
from capturecheck.checks.coverage import CameraPath
from capturecheck.degrade import ops
from capturecheck.degrade.make_variants import make_all, variant_id
from capturecheck.io.dataset import list_captures


def mean_sharpness(frames) -> float:
    return float(np.mean([quality.sharpness(f) for f in frames]))


def test_blur_lowers_mean_sharpness(trans_frames):
    result = ops.blur(trans_frames, ratio=1.0, sigma=9)
    assert result.n == len(trans_frames)
    assert mean_sharpness(result.frames) < mean_sharpness(trans_frames)
    assert result.labels["motion_blur"] == 1
    assert len(result.labels["frames"]) == len(trans_frames)


def test_blur_only_touches_its_block(trans_frames):
    result = ops.blur(trans_frames, ratio=0.25, sigma=9, rng=np.random.default_rng(1))
    affected = set(result.labels["frames"])
    assert 0 < len(affected) < len(trans_frames)
    for i, (before, after) in enumerate(zip(trans_frames, result.frames, strict=True)):
        if i in affected:
            assert quality.sharpness(after) < quality.sharpness(before)
        else:
            assert np.array_equal(before, after)


def test_subsample_returns_the_right_count(trans_frames):
    result = ops.subsample(trans_frames, keep_ratio=0.5)
    assert result.n == round(0.5 * len(trans_frames))
    assert result.labels["too_few_views"] == 1
    assert len(result.names) == result.n and len(result.times) == result.n


def test_rotation_only_keeps_a_contiguous_stretch(trans_frames):
    result = ops.rotation_only(trans_frames, keep_ratio=0.4)
    assert result.labels["no_walking"] == 1
    assert result.n == max(2, round(0.4 * len(trans_frames)))
    start, end = result.labels["kept"]
    assert end - start + 1 == result.n


def test_exposure_jitter_changes_brightness_frame_by_frame(trans_frames):
    result = ops.exposure_jitter(trans_frames, strength=0.6, rng=np.random.default_rng(2))
    before = [quality.brightness(f) for f in trans_frames]
    after = [quality.brightness(f) for f in result.frames]
    assert result.labels["lighting"] == 1
    assert np.std(after) > np.std(before)


def test_noise_raises_the_noise_estimate(trans_frames):
    result = ops.noise(trans_frames, sigma=20, jpeg_quality=30)
    before = float(np.mean([quality.noise_level(f) for f in trans_frames]))
    after = float(np.mean([quality.noise_level(f) for f in result.frames]))
    assert after > before
    assert result.labels["noisy"] == 1


def test_drop_sector_without_a_path_removes_a_block(trans_frames):
    result = ops.drop_sector(trans_frames, path=None, sector_deg=90)
    assert result.labels["poor_coverage"] == 1
    assert result.n < len(trans_frames)
    assert result.labels["removed"] == len(trans_frames) - result.n


def test_drop_sector_with_a_path_removes_one_direction():
    """A synthetic orbit: a 90-degree sector is a quarter of the frames."""
    n = 24
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    centers = np.stack([np.sin(angles), np.zeros(n), np.cos(angles)], axis=1) * 5.0
    path = CameraPath(centers=centers, rotations=np.tile(np.eye(3), (n, 1, 1)))
    frames = [np.full((32, 32, 3), i * 4, dtype=np.uint8) for i in range(n)]
    result = ops.drop_sector(frames, path=path, sector_deg=90, rng=np.random.default_rng(0))
    assert result.labels["poor_coverage"] == 1
    assert result.labels["removed"] == n // 4
    assert result.labels["side"] in {
        "front", "front-right", "right", "back-right",
        "back", "back-left", "left", "front-left",
    }


def test_drop_gap_removes_neighbouring_photos(bundle_frames):
    names = [f"IMG_{i:04d}.jpg" for i in range(len(bundle_frames))]
    result = ops.drop_gap(bundle_frames, n=4, names=names)
    assert result.labels["low_overlap"] == 1
    assert len(result.labels["photos"]) == 4
    assert result.n == len(bundle_frames) - 4
    # The removed names are gone and the rest keep their order.
    assert not set(result.labels["photos"]) & set(result.names)
    assert result.names == sorted(result.names)


def test_every_op_keeps_names_and_times_aligned(bundle_frames):
    names = [f"IMG_{i:04d}.jpg" for i in range(len(bundle_frames))]
    times = [float(i) for i in range(len(bundle_frames))]
    for name, function in ops.OPS.items():
        result = function(bundle_frames, names=names, times=times)
        assert result.n == len(result.names) == len(result.times), name
        assert result.labels, name


def test_make_all_writes_copies_with_exact_labels(tmp_path, tiny_capture_dir, tiny_bundle):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "kitchen_video").symlink_to(tiny_capture_dir)
    (raw / "kitchen_photos").symlink_to(tiny_bundle)
    out = tmp_path / "degraded"

    variants = make_all(raw_dir=raw, out_dir=out, strengths=["medium"])
    by_id = {v.capture_id: v for v in variants}

    # The 20-photo bundle supports every op, drop_gap included.
    bundle_ops = {v.op for v in variants if v.source_id == "kitchen_photos"}
    assert bundle_ops == set(ops.OPS)
    # drop_gap is bundle-only, and the 4-frame test video is too short to thin out.
    video_ops = {v.op for v in variants if v.source_id == "kitchen_video"}
    assert "drop_gap" not in video_ops
    assert video_ops.issubset(set(ops.OPS) - {"drop_gap"})
    assert variant_id("kitchen_photos", "drop_gap", "medium") in by_id

    blur = by_id[variant_id("kitchen_photos", "blur", "medium")]
    assert blur.scene_id == "kitchen_photos"
    payload = json.loads((blur.path / "labels.json").read_text())
    assert payload["motion_blur"] == 1
    assert payload["degrade_type"] == "blur"
    assert (blur.path / "scene.txt").read_text().strip() == "kitchen_photos"
    assert len(list((blur.path / "images").glob("*.jpg"))) == blur.n_frames

    # The copies are discoverable, and marked as degraded.
    refs = list_captures(out)
    assert len(refs) == len(variants)
    assert all(r.source == "degraded" for r in refs)
    assert all(r.input_type == "images" for r in refs)

    # The originals were not touched.
    assert (tiny_capture_dir / "video.mp4").is_file()
    assert len(list(tiny_bundle.glob("*.jpg"))) == 20


def test_make_all_skips_company_captures(tmp_path, tiny_bundle):
    raw = tmp_path / "raw"
    folder = raw / "company_scan"
    folder.mkdir(parents=True)
    for photo in sorted(tiny_bundle.glob("*.jpg"))[:6]:
        (folder / photo.name).write_bytes(photo.read_bytes())
    (folder / "source.txt").write_text("company\n")

    variants = make_all(raw_dir=raw, out_dir=tmp_path / "degraded", strengths=["light"])
    assert variants == []
