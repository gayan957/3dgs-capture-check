"""Break a good capture on purpose, and know exactly what you broke.

Every op takes a list of frames (with their names and timestamps) and returns new
frames plus the labels that are now true by construction. These copies are the
cheapest training data we have: the label needs no human and no 3DGS run.

Ops work the same on video frames and on photos. ``drop_gap`` is the one
exception and is only meaningful for photo bundles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from capturecheck.checks.coverage import CameraPath
from capturecheck.checks.motion import overlap

#: Share of the kept pairs that must read "turned" before ``rotation_only`` may
#: claim the capture shows no walking.
TURNED_SHARE_FOR_NO_WALKING = 0.5


@dataclass
class DegradeResult:
    """New frames (with their names and times) and the labels that now hold."""

    frames: list[np.ndarray]
    names: list[str]
    times: list[float | None]
    labels: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.frames)


def _names_times(
    frames: list[np.ndarray],
    names: list[str] | None,
    times: list[float | None] | None,
) -> tuple[list[str], list[float | None]]:
    names = list(names) if names is not None else [f"frame_{i:03d}" for i in range(len(frames))]
    times = list(times) if times is not None else [None] * len(frames)
    return names, times


def _motion_blur_kernel(size: int, angle_deg: float) -> np.ndarray:
    """A line kernel: convolving with it smears the image along one direction."""
    size = max(3, int(size) | 1)  # odd, at least 3
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = 1.0
    centre = (size / 2 - 0.5, size / 2 - 0.5)
    rotation = cv2.getRotationMatrix2D(centre, float(angle_deg), 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size))
    total = kernel.sum()
    return kernel / total if total > 0 else kernel


def blur(
    frames: list[np.ndarray],
    ratio: float = 0.3,
    sigma: float = 7.0,
    names: list[str] | None = None,
    times: list[float | None] | None = None,
    rng: np.random.Generator | None = None,
) -> DegradeResult:
    """Smear a contiguous block of frames, as a hand shake or a fast pan would."""
    rng = rng or np.random.default_rng(0)
    names, times = _names_times(frames, names, times)
    n = len(frames)
    count = int(round(np.clip(ratio, 0.0, 1.0) * n))
    out = [f.copy() for f in frames]
    if count <= 0 or n == 0:
        return DegradeResult(out, names, times, {"motion_blur": 0, "frames": []})

    start = int(rng.integers(0, max(1, n - count + 1)))
    affected = list(range(start, min(n, start + count)))
    size = max(3, int(round(sigma * 2 + 1)))
    kernel = _motion_blur_kernel(size, float(rng.uniform(0, 180)))
    for i in affected:
        out[i] = cv2.filter2D(out[i], -1, kernel)

    return DegradeResult(
        out, names, times,
        {
            "motion_blur": 1,
            "frames": affected,
            "where": f"{names[affected[0]]} to {names[affected[-1]]}",
        },
    )


def drop_sector(
    frames: list[np.ndarray],
    path: CameraPath | None = None,
    sector_deg: float = 90.0,
    names: list[str] | None = None,
    times: list[float | None] | None = None,
    rng: np.random.Generator | None = None,
) -> DegradeResult:
    """Delete the frames shot from one direction, leaving a hole in the coverage.

    With a camera path, the direction is the real azimuth of each camera. Without
    one, a contiguous block of the same share is removed instead, which has the
    same effect on coverage for a capture that walks around its subject.
    """
    from capturecheck.checks.coverage import AZIMUTH_NAMES

    rng = rng or np.random.default_rng(0)
    names, times = _names_times(frames, names, times)
    n = len(frames)
    if n == 0:
        return DegradeResult([], [], [], {"poor_coverage": 0})

    share = float(np.clip(sector_deg / 360.0, 0.0, 0.9))
    keep = np.ones(n, dtype=bool)
    side = "unknown"

    usable_path = path is not None and path.n >= n and path.extent > 1e-6
    if usable_path:
        centre = path.centers[:n].mean(axis=0)
        vectors = path.centers[:n] - centre
        azimuth = np.degrees(np.arctan2(vectors[:, 0], vectors[:, 2])) % 360.0
        start = float(rng.uniform(0, 360))
        offset = (azimuth - start) % 360.0
        keep = offset > sector_deg
        if keep.sum() < 3:  # never delete nearly everything
            keep = np.ones(n, dtype=bool)
            usable_path = False
        else:
            middle = (start + sector_deg / 2.0) % 360.0
            side = AZIMUTH_NAMES[int(middle // (360 / len(AZIMUTH_NAMES)))]

    if not usable_path:
        count = max(1, int(round(share * n)))
        start_idx = int(rng.integers(0, max(1, n - count + 1)))
        keep[start_idx : start_idx + count] = False
        side = "front" if start_idx < n / 3 else ("back" if start_idx > 2 * n / 3 else "middle")

    kept = np.where(keep)[0]
    return DegradeResult(
        [frames[i].copy() for i in kept],
        [names[i] for i in kept],
        [times[i] for i in kept],
        {
            "poor_coverage": 1,
            "side": side,
            "removed": int(n - kept.size),
            "sector_deg": float(sector_deg),
        },
    )


def rotation_only(
    frames: list[np.ndarray],
    keep_ratio: float = 0.4,
    pairs: list[tuple[int, int, object]] | None = None,
    names: list[str] | None = None,
    times: list[float | None] | None = None,
    rng: np.random.Generator | None = None,
) -> DegradeResult:
    """Keep only the stretch where the camera turned instead of walking.

    This op can only *select*; it cannot manufacture a pivot. A capture that
    walks the whole way round has no rotation-only stretch to keep, and keeping
    a short contiguous window of it still shows the camera walking. In that case
    the result carries ``no_walking: 0`` and ``reason``, and the caller should
    discard it rather than write a copy whose headline label is false: a wrong
    label teaches the model the opposite of the truth, which is worse than one
    fewer training example.
    """
    from capturecheck.checks.motion import TURNED, WALKED

    rng = rng or np.random.default_rng(0)
    names, times = _names_times(frames, names, times)
    n = len(frames)
    if n == 0:
        return DegradeResult([], [], [], {"no_walking": 0, "reason": "no frames"})

    window = max(2, int(round(np.clip(keep_ratio, 0.05, 1.0) * n)))

    best_start = None
    turned_share = float("nan")
    if pairs:
        turned = np.zeros(n, dtype=float)
        decided = np.zeros(n, dtype=float)
        for i, j, result in pairs:
            kind = getattr(result, "kind", None)
            if kind not in (TURNED, WALKED):
                continue
            for idx in (i, j):
                if 0 <= idx < n:
                    decided[idx] += 1.0
                    turned[idx] += 1.0 if kind == TURNED else 0.0
        if decided.sum() > 0:
            sums = np.convolve(turned, np.ones(window), mode="valid")
            best_start = int(np.argmax(sums))
            kept_slice = slice(best_start, best_start + window)
            kept_decided = decided[kept_slice].sum()
            turned_share = (
                float(turned[kept_slice].sum() / kept_decided) if kept_decided else float("nan")
            )

    if best_start is None:
        best_start = int(rng.integers(0, max(1, n - window + 1)))

    kept = list(range(best_start, min(n, best_start + window)))
    labels: dict = {"kept": [kept[0], kept[-1]], "n_kept": len(kept)}

    if np.isfinite(turned_share) and turned_share >= TURNED_SHARE_FOR_NO_WALKING:
        labels["no_walking"] = 1
        labels["turned_share"] = round(turned_share, 3)
    elif np.isfinite(turned_share):
        labels["no_walking"] = 0
        labels["turned_share"] = round(turned_share, 3)
        labels["reason"] = (
            f"only {turned_share:.0%} of the kept pairs turned: this capture has no "
            "rotation-only stretch to cut down to"
        )
    else:
        # No pair decisions to go on: a shortened capture does have less
        # baseline, but calling that "no walking" would be a guess.
        labels["no_walking"] = 0
        labels["reason"] = "no walked-or-turned decisions were available"

    return DegradeResult(
        [frames[i].copy() for i in kept],
        [names[i] for i in kept],
        [times[i] for i in kept],
        labels,
    )


def subsample(
    frames: list[np.ndarray],
    keep_ratio: float = 0.25,
    names: list[str] | None = None,
    times: list[float | None] | None = None,
    rng: np.random.Generator | None = None,
) -> DegradeResult:
    """Thin the capture out evenly: fewer viewpoints, same path."""
    names, times = _names_times(frames, names, times)
    n = len(frames)
    count = max(1, int(round(np.clip(keep_ratio, 0.0, 1.0) * n)))
    if n == 0:
        return DegradeResult([], [], [], {"too_few_views": 0})
    kept = np.unique(np.linspace(0, n - 1, num=count).round().astype(int))
    return DegradeResult(
        [frames[i].copy() for i in kept],
        [names[i] for i in kept],
        [times[i] for i in kept],
        {"too_few_views": 1, "n_kept": int(kept.size), "n_before": n},
    )


def exposure_jitter(
    frames: list[np.ndarray],
    strength: float = 0.35,
    names: list[str] | None = None,
    times: list[float | None] | None = None,
    rng: np.random.Generator | None = None,
) -> DegradeResult:
    """Per-frame gain, gamma and white-balance wobble: auto-exposure gone wrong."""
    rng = rng or np.random.default_rng(0)
    names, times = _names_times(frames, names, times)
    strength = float(np.clip(strength, 0.0, 1.0))
    out: list[np.ndarray] = []
    for frame in frames:
        gain = float(1.0 + rng.normal(0.0, 0.5 * strength))
        gamma = float(np.exp(rng.normal(0.0, 0.5 * strength)))
        balance = 1.0 + rng.normal(0.0, 0.3 * strength, size=3)
        img = frame.astype(np.float32) / 255.0
        img = np.clip(img * max(0.05, gain) * balance.reshape(1, 1, 3), 0.0, 1.0)
        img = np.power(img, np.clip(gamma, 0.2, 5.0))
        out.append((img * 255.0).astype(np.uint8))
    return DegradeResult(out, names, times, {"lighting": 1, "strength": strength})


def noise(
    frames: list[np.ndarray],
    sigma: float = 14.0,
    jpeg_quality: int = 35,
    names: list[str] | None = None,
    times: list[float | None] | None = None,
    rng: np.random.Generator | None = None,
) -> DegradeResult:
    """Add sensor noise and squash the frames through heavy JPEG."""
    rng = rng or np.random.default_rng(0)
    names, times = _names_times(frames, names, times)
    out: list[np.ndarray] = []
    for frame in frames:
        noisy = frame.astype(np.float32) + rng.normal(0.0, float(sigma), frame.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        ok, buffer = cv2.imencode(
            ".jpg", noisy, [int(cv2.IMWRITE_JPEG_QUALITY), int(np.clip(jpeg_quality, 5, 100))]
        )
        out.append(cv2.imdecode(buffer, cv2.IMREAD_COLOR) if ok else noisy)
    return DegradeResult(
        out, names, times,
        {"noisy": 1, "sigma": float(sigma), "jpeg_quality": int(jpeg_quality)},
    )


def drop_gap(
    frames: list[np.ndarray],
    share: float = 0.4,
    photos_per_gap: int = 2,
    min_overlap: float | None = None,
    max_photos_per_gap: int = 6,
    features: list | None = None,
    names: list[str] | None = None,
    times: list[float | None] | None = None,
    rng: np.random.Generator | None = None,
) -> DegradeResult:
    """Punch gaps into a bundle until `low_overlap` is actually true of it.

    Image bundles only. A video cannot have this defect at 2 fps sampling without
    also losing coverage, which :func:`drop_sector` already covers.

    One gap is not this defect. The rule reads `low_overlap` off
    `overlap_low_ratio`, the *share* of neighbouring pairs that match too
    little, so a single gap in a 144-photo orbit moves it to 0.007 against a
    threshold of 0.2, and the copy carries a label that is not true of it. This
    cuts in enough separate places for the share of broken junctions to reach
    `share`, which the config keeps above the rule's threshold.

    Given `features` (ORB per photo, which `make_variants` already has), every
    junction is *measured*: the gap widens until the two photos left facing each
    other really do match below `min_overlap`, up to `max_photos_per_gap`. Gaps
    that will not break are dropped rather than counted, because a junction that
    still overlaps is not a gap. Without `features` the cut falls back to
    `photos_per_gap` photos and the share is a geometric estimate.

    A bundle too short to hold the gaps, or one whose photos stay matched
    however wide the cut, cannot have the defect, so it returns
    ``low_overlap: 0`` and a ``reason`` instead of a mislabelled copy.
    """
    rng = rng or np.random.default_rng(0)
    names, times = _names_times(frames, names, times)
    total = len(frames)
    per_gap = max(1, int(photos_per_gap))
    widest = max(per_gap, int(max_photos_per_gap))
    target = float(np.clip(share, 0.0, 1.0))

    from capturecheck.config import get_config

    overlap_cfg = get_config().labeling.overlap
    if min_overlap is None:
        min_overlap = overlap_cfg.min_overlap
    # The share the rule itself needs before it calls this low overlap.
    rule_ratio = overlap_cfg.low_overlap_ratio

    def refuse(reason: str) -> DegradeResult:
        return DegradeResult(
            [f.copy() for f in frames], names, times,
            {"low_overlap": 0, "photos": [], "reason": reason},
        )

    # Each gap leaves one junction below the threshold, so after G gaps the ratio
    # is G / (kept - 1) = G / (total - G*per_gap - 1). The smallest G that reaches
    # `target` is:
    wanted = (
        int(np.ceil(target * (total - 1) / (1.0 + target * per_gap))) if total > 1 else 0
    )
    # Every gap needs a kept photo on each side: G gaps need G*per_gap removed and
    # G + 1 kept.
    room = (total - 1) // (per_gap + 1)
    gaps = min(wanted, room)
    if gaps < 1 or target <= 0:
        return refuse(
            f"{total} photos have room for {room} gaps of {per_gap}; "
            f"a share of {target:.2f} of pairs needs {wanted}"
        )

    def breaks(left: int, right: int) -> bool:
        """Do the two photos left facing each other match too little?"""
        if features is None or not (0 <= left < len(features) and 0 <= right < len(features)):
            return True  # unverified: the geometric estimate stands in
        return overlap(features[left], features[right]) < float(min_overlap)

    # Spread the cuts over the bundle, jittered inside each slot so copies are not
    # all cut on the same grid, and widen each one until its junction really goes.
    stride = total / float(gaps)
    removed: list[int] = []
    taken: set[int] = set()
    unbroken = 0
    for g in range(gaps):
        slot_start = int(g * stride) + 1
        slot_end = min(int((g + 1) * stride), total - 1) - per_gap
        if slot_end < slot_start:
            continue
        begin = (
            int(rng.integers(slot_start, slot_end + 1)) if slot_end > slot_start else slot_start
        )
        width = next(
            (
                w
                for w in range(per_gap, widest + 1)
                if begin + w <= total - 1 and breaks(begin - 1, begin + w)
            ),
            None,
        )
        if width is None:
            unbroken += 1
            continue
        run = list(range(begin, begin + width))
        # Keep a photo between this gap and the last, or the two merge into one
        # wider gap and break a single junction instead of two.
        if any(i in taken or i - 1 in taken for i in run):
            continue
        taken.update(run)
        removed.extend(run)

    if not removed:
        return refuse(
            f"no gap in these {total} photos drops a junction below {float(min_overlap):.2f} "
            f"overlap, even {widest} photos wide"
        )

    removed_set = set(removed)
    kept = [i for i in range(total) if i not in removed_set]
    n_gaps = len(removed) - sum(1 for i in removed if i - 1 in removed_set)
    achieved = n_gaps / float(max(1, len(kept) - 1))
    if features is not None and achieved <= rule_ratio:
        return refuse(
            f"only {n_gaps} of {gaps} gaps broke a junction, a share of {achieved:.3f}; "
            "that is not low overlap"
        )

    first = min(removed)
    return DegradeResult(
        [frames[i].copy() for i in kept],
        [names[i] for i in kept],
        [times[i] for i in kept],
        {
            "low_overlap": 1,
            "photos": [names[i] for i in sorted(removed)],
            "gaps": n_gaps,
            "overlap_low_ratio_expected": round(achieved, 4),
            "unbroken_gaps": unbroken,
            "gap_after": names[first - 1] if first > 0 else names[0],
        },
    )


def defect_created(labels: dict) -> bool:
    """Did the op actually produce the problem it exists to produce?

    Every op sets exactly one capture label. When that label comes back 0 the op
    could not create the defect on this capture, and the copy should be thrown
    away rather than written with a label that is not true of it.
    """
    from capturecheck.labeling import schema

    present = [k for k in labels if k in schema.VIDEO_LABEL_NAMES]
    return bool(present) and any(int(labels[k]) == 1 for k in present)


#: Name -> function, so ``make_variants`` and the tests can look ops up by name.
OPS: dict[str, Callable[..., DegradeResult]] = {
    "blur": blur,
    "drop_sector": drop_sector,
    "rotation_only": rotation_only,
    "subsample": subsample,
    "exposure_jitter": exposure_jitter,
    "noise": noise,
    "drop_gap": drop_gap,
}

#: Ops that only make sense for photo bundles.
IMAGES_ONLY_OPS: tuple[str, ...] = ("drop_gap",)
