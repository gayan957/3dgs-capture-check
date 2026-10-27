"""Broken copies: each op must change the right measurement and label it exactly."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

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
    assert result.n == max(2, round(0.4 * len(trans_frames)))
    start, end = result.labels["kept"]
    assert end - start + 1 == result.n
    # Without walked-or-turned decisions it cannot claim the capture stopped
    # walking, so it says so rather than guessing.
    assert result.labels["no_walking"] == 0
    assert "no walked-or-turned decisions" in result.labels["reason"]


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
    result = ops.drop_gap(bundle_frames, share=0.4, names=names)
    assert result.labels["low_overlap"] == 1
    assert result.n == len(bundle_frames) - len(result.labels["photos"])
    # The removed names are gone and the rest keep their order.
    assert not set(result.labels["photos"]) & set(result.names)
    assert result.names == sorted(result.names)


def test_drop_gap_makes_enough_gaps_to_earn_the_label(bundle_frames):
    """One gap is not low overlap: the rule needs a *share* of bad pairs.

    `low_overlap` fires on `overlap_low_ratio > 0.2`, so a copy with a single
    gap in a long bundle carries a label that is not true of it. The op has to
    cut in enough places to clear the rule's own threshold.
    """
    from capturecheck.config import get_config

    rule_ratio = get_config().labeling.overlap.low_overlap_ratio
    names = [f"IMG_{i:04d}.jpg" for i in range(len(bundle_frames))]

    result = ops.drop_gap(bundle_frames, share=0.4, names=names)
    assert result.labels["gaps"] > 1, "a single gap cannot make this label true"
    assert result.labels["overlap_low_ratio_expected"] > rule_ratio


def test_drop_gap_strengths_are_ordered(bundle_frames):
    names = [f"IMG_{i:04d}.jpg" for i in range(len(bundle_frames))]
    ratios = [
        ops.drop_gap(bundle_frames, share=share, names=names)
        .labels["overlap_low_ratio_expected"]
        for share in (0.25, 0.4, 0.6)
    ]
    assert ratios == sorted(ratios)


def test_drop_gap_refuses_a_bundle_too_short_to_have_the_defect():
    """Rule 10: no copy rather than a copy with a false label."""
    frames = [np.full((32, 32, 3), 120, np.uint8) for _ in range(3)]
    result = ops.drop_gap(frames, share=0.6, photos_per_gap=2, names=None)
    assert result.labels["low_overlap"] == 0
    assert "reason" in result.labels
    assert not ops.defect_created(result.labels)


def test_drop_gap_refuses_a_bundle_whose_photos_stay_matched(bundle_frames):
    """`bundle_frames` pans 9 px a photo; no cut can make its photos strangers."""
    from capturecheck.checks.motion import orb_frames

    names = [f"IMG_{i:04d}.jpg" for i in range(len(bundle_frames))]
    result = ops.drop_gap(
        bundle_frames, share=0.4, names=names, features=orb_frames(bundle_frames)
    )
    assert result.labels["low_overlap"] == 0
    assert "reason" in result.labels
    assert result.n == len(bundle_frames)


def test_drop_gap_refuses_when_no_gap_breaks_a_junction(bundle_frames):
    """A junction that still matches is not a gap, however wide the cut."""

    class AlwaysMatches:
        """ORB stand-in whose photos always overlap: nothing can break them."""

        n = 100
        shape = (64, 64)

    names = [f"IMG_{i:04d}.jpg" for i in range(len(bundle_frames))]
    features = [AlwaysMatches() for _ in bundle_frames]
    # overlap() on these would be 1.0, so every candidate junction survives.
    import capturecheck.degrade.ops as ops_module

    original = ops_module.overlap
    ops_module.overlap = lambda a, b: 1.0
    try:
        result = ops.drop_gap(bundle_frames, share=0.4, names=names, features=features)
    finally:
        ops_module.overlap = original

    assert result.labels["low_overlap"] == 0
    assert "reason" in result.labels
    assert result.n == len(bundle_frames), "a refused op must not change the capture"


def test_drop_gap_junctions_really_fall_below_the_threshold(wide_pan_frames):
    """With ORB features in hand, every gap it keeps is a measured gap.

    This is the assertion rule 10 exists for: relabel the copy from its pixels
    and the measurement must agree with the label it was written with.
    """
    from capturecheck.checks.motion import orb_frames, overlap_series
    from capturecheck.config import get_config

    cfg = get_config().labeling.overlap
    names = [f"IMG_{i:04d}.jpg" for i in range(len(wide_pan_frames))]
    features = orb_frames(wide_pan_frames)
    result = ops.drop_gap(wide_pan_frames, share=0.4, names=names, features=features)
    assert result.labels["low_overlap"] == 1, result.labels.get("reason")

    measured = np.array(overlap_series(orb_frames(result.frames)), dtype=float)
    got = float((measured < cfg.min_overlap).mean())
    assert got > cfg.low_overlap_ratio, (
        f"copy labelled low_overlap measures {got:.3f}, under the rule's {cfg.low_overlap_ratio}"
    )


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

    # This bundle pans only 9 px a photo, so its photos stay matched however wide
    # the cut: it cannot have low overlap, and no copy claims it does (rule 10).
    bundle_ops = {v.op for v in variants if v.source_id == "kitchen_photos"}
    assert bundle_ops == set(ops.OPS) - {"drop_gap"}
    assert variant_id("kitchen_photos", "drop_gap", "medium") not in by_id
    # drop_gap is bundle-only, and the 4-frame test video is too short to thin out.
    video_ops = {v.op for v in variants if v.source_id == "kitchen_video"}
    assert "drop_gap" not in video_ops
    assert video_ops.issubset(set(ops.OPS) - {"drop_gap"})

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


def test_make_all_writes_a_drop_gap_copy_when_the_bundle_can_have_one(
    tmp_path, wide_pan_bundle
):
    """The other side of rule 10: a bundle that *can* have the defect gets one."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "hallway_photos").symlink_to(wide_pan_bundle)

    variants = make_all(raw_dir=raw, out_dir=tmp_path / "degraded", strengths=["medium"])
    by_id = {v.capture_id: v for v in variants}
    vid = variant_id("hallway_photos", "drop_gap", "medium")
    assert vid in by_id, "a bundle with real parallax should support drop_gap"

    payload = json.loads((by_id[vid].path / "labels.json").read_text())
    assert payload["low_overlap"] == 1
    assert payload["gaps"] > 1, "one gap is not low overlap"


def test_make_all_skips_company_captures(tmp_path, tiny_bundle):
    raw = tmp_path / "raw"
    folder = raw / "company_scan"
    folder.mkdir(parents=True)
    for photo in sorted(tiny_bundle.glob("*.jpg"))[:6]:
        (folder / photo.name).write_bytes(photo.read_bytes())
    (folder / "source.txt").write_text("company\n")

    variants = make_all(raw_dir=raw, out_dir=tmp_path / "degraded", strengths=["light"])
    assert variants == []


def test_noise_strengths_are_ordered(trans_frames):
    """Heavy must measure noisier than medium: JPEG used to erase the difference."""
    from capturecheck.config import get_config

    cfg = get_config()
    measured = []
    for strength in ("light", "medium", "heavy"):
        params = cfg.degrade.params("noise", strength)
        result = ops.noise(trans_frames, rng=np.random.default_rng(0), **params)
        measured.append(float(np.mean([quality.noise_level(f) for f in result.frames])))
    assert measured[0] < measured[1] < measured[2], measured


def test_rotation_only_refuses_to_mislabel_a_walking_capture(trans_frames):
    """It can select a stretch, not manufacture a pivot."""
    from capturecheck.checks import motion

    pairs = motion.motion_pairs(trans_frames, gap=1, adaptive=False)
    assert motion.walked_ratio(pairs) > 0.75  # every pair walks

    result = ops.rotation_only(trans_frames, keep_ratio=0.4, pairs=pairs)
    assert result.labels["no_walking"] == 0
    assert "no rotation-only stretch" in result.labels["reason"]
    assert not ops.defect_created(result.labels)


def test_rotation_only_labels_a_real_pivot(rot_frames):
    from capturecheck.checks import motion

    pairs = motion.motion_pairs(rot_frames, gap=1, adaptive=False)
    assert motion.walked_ratio(pairs) < 0.25  # every pair turns

    result = ops.rotation_only(rot_frames, keep_ratio=0.5, pairs=pairs)
    assert result.labels["no_walking"] == 1
    assert result.labels["turned_share"] >= 0.5
    assert ops.defect_created(result.labels)


def test_defect_created():
    assert ops.defect_created({"motion_blur": 1, "frames": [1, 2]})
    assert not ops.defect_created({"no_walking": 0, "reason": "nothing to cut"})
    assert not ops.defect_created({"kept": [0, 5]})  # no capture label at all


def test_make_all_skips_variants_whose_defect_failed(tmp_path, tiny_bundle, caplog):
    """A copy with a false headline label is worse than one fewer example."""
    import logging

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "kitchen_photos").symlink_to(tiny_bundle)

    with caplog.at_level(logging.INFO):
        variants = make_all(
            raw_dir=raw, out_dir=tmp_path / "degraded",
            ops=["rotation_only"], strengths=["medium"],
        )
    for variant in variants:
        assert variant.labels.get("no_walking") == 1
    if not variants:
        assert "could not create its defect" in caplog.text


def test_the_same_broken_copy_twice_is_byte_identical(tmp_path, tiny_bundle):
    """Two `degrade` runs in separate processes must produce the same pixels.

    The copies are training data, so a copy that differs between runs means the
    model trained on something nobody can reproduce. Separate processes are the
    whole point: the seed used to come from `hash()` of the capture id, which is
    salted per interpreter, so a single process could never see the bug.
    """
    import hashlib
    import subprocess
    import sys

    raw = tmp_path / "raw"
    raw.mkdir()
    shutil.copytree(tiny_bundle, raw / "kitchen_photos")
    (raw / "kitchen_photos" / "scene.txt").write_text("kitchen\n")

    def run(out: Path, hash_seed: str) -> str:
        subprocess.run(
            [
                sys.executable, "-m", "capturecheck.cli", "-q", "degrade",
                "--raw", str(raw), "--out", str(out), "--op", "noise",
            ],
            check=True,
            capture_output=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        digest = hashlib.sha1()
        for photo in sorted(out.rglob("*.jpg")):
            digest.update(photo.read_bytes())
        return digest.hexdigest()

    first = run(tmp_path / "a", "0")
    second = run(tmp_path / "b", "12345")
    assert first == second
    assert first != hashlib.sha1().hexdigest(), "no copies were written at all"
