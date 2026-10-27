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

    With pair decisions available, the densest run of "turned" pairs is kept.
    Otherwise a short contiguous stretch is kept, which has the same effect: too
    little baseline for 3DGS to recover depth.
    """
    from capturecheck.checks.motion import TURNED

    rng = rng or np.random.default_rng(0)
    names, times = _names_times(frames, names, times)
    n = len(frames)
    if n == 0:
        return DegradeResult([], [], [], {"no_walking": 0})

    window = max(2, int(round(np.clip(keep_ratio, 0.05, 1.0) * n)))

    best_start = None
    if pairs:
        turned = np.zeros(n, dtype=float)
        for i, j, result in pairs:
            weight = 1.0 if getattr(result, "kind", None) == TURNED else 0.0
            for idx in (i, j):
                if 0 <= idx < n:
                    turned[idx] += weight
        if turned.sum() > 0:
            sums = np.convolve(turned, np.ones(window), mode="valid")
            best_start = int(np.argmax(sums))

    if best_start is None:
        best_start = int(rng.integers(0, max(1, n - window + 1)))

    kept = list(range(best_start, min(n, best_start + window)))
    return DegradeResult(
        [frames[i].copy() for i in kept],
        [names[i] for i in kept],
        [times[i] for i in kept],
        {"no_walking": 1, "kept": [kept[0], kept[-1]], "n_kept": len(kept)},
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
    n: int = 5,
    names: list[str] | None = None,
    times: list[float | None] | None = None,
    rng: np.random.Generator | None = None,
) -> DegradeResult:
    """Remove ``n`` neighbouring photos: the classic "I stopped shooting here" gap.

    Image bundles only. A video cannot have this defect at 2 fps sampling without
    also losing coverage, which :func:`drop_sector` already covers.
    """
    rng = rng or np.random.default_rng(0)
    names, times = _names_times(frames, names, times)
    total = len(frames)
    count = int(np.clip(n, 0, max(0, total - 3)))
    if total == 0 or count <= 0:
        return DegradeResult(
            [f.copy() for f in frames], names, times, {"low_overlap": 0, "photos": []}
        )
    # Keep at least one photo on each side of the gap.
    start = int(rng.integers(1, max(2, total - count)))
    removed = list(range(start, start + count))
    kept = [i for i in range(total) if i not in set(removed)]
    return DegradeResult(
        [frames[i].copy() for i in kept],
        [names[i] for i in kept],
        [times[i] for i in kept],
        {
            "low_overlap": 1,
            "photos": [names[i] for i in removed],
            "gap_after": names[removed[0] - 1] if removed[0] > 0 else names[0],
        },
    )


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
