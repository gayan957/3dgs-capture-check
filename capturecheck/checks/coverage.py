"""A rough camera path, and which directions around the subject were filmed.

This is not SfM: relative poses from consecutive essential matrices are chained
together with an unknown (and drifting) scale. That is enough to answer the one
question we care about -- "did the camera ever get to the other side?" -- and it
is why every number here is reported as approximate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from capturecheck.checks.motion import (
    OrbFrame,
    PairMotion,
    approx_intrinsics,
    orb_frames,
    walked_or_turned,
)
from capturecheck.config import get_config

#: Names for the 8 azimuth bins, used in the evidence strings.
AZIMUTH_NAMES = (
    "front", "front-right", "right", "back-right",
    "back", "back-left", "left", "front-left",
)
ELEVATION_NAMES = ("low", "eye level", "high")


@dataclass
class CameraPath:
    """Approximate camera centres and orientations, in an arbitrary frame and scale."""

    centers: np.ndarray  # (N, 3)
    rotations: np.ndarray  # (N, 3, 3) world-from-camera
    reliable: bool = True
    note: str = ""
    n_steps_used: int = 0

    @property
    def n(self) -> int:
        return int(self.centers.shape[0])

    @property
    def extent(self) -> float:
        """Largest distance between two camera centres (in the unknown unit)."""
        if self.n < 2:
            return 0.0
        spread = self.centers.max(axis=0) - self.centers.min(axis=0)
        return float(np.linalg.norm(spread))


@dataclass
class Coverage:
    """Which directions around the subject the camera visited."""

    grid: np.ndarray  # (azimuth_bins, elevation_bins) counts
    empty_bins: int  # empty bins in the occupied elevation rows
    empty_bins_all: int  # empty bins in the whole grid
    empty_names: list[str] = field(default_factory=list)
    azimuth_span_deg: float = 0.0
    reliable: bool = True

    @property
    def filled_bins(self) -> int:
        return int((self.grid > 0).sum())

    def as_dict(self) -> dict:
        return {
            "empty_bins": self.empty_bins,
            "empty_bins_all": self.empty_bins_all,
            "filled_bins": self.filled_bins,
            "azimuth_span_deg": round(self.azimuth_span_deg, 1),
            "empty_names": self.empty_names,
            "reliable": self.reliable,
        }


def rough_path(
    frames: list[np.ndarray] | None = None,
    features: list[OrbFrame] | None = None,
    pairs: list[tuple[int, int, PairMotion]] | None = None,
    focal_px: float | None = None,
    step: int = 1,
) -> CameraPath:
    """Chain relative poses from consecutive frames into an approximate path.

    Pass ``pairs`` from :func:`capturecheck.checks.motion.motion_pairs` to reuse
    work already done; otherwise the poses are estimated here.
    """
    if pairs is None:
        if features is None:
            if not frames:
                return CameraPath(
                    centers=np.zeros((0, 3)), rotations=np.zeros((0, 3, 3)),
                    reliable=False, note="no frames",
                )
            features = orb_frames(frames)
        k = approx_intrinsics(features[0].shape, focal_px)
        pairs = [
            (i, i + step, walked_or_turned(features[i], features[i + step], K=k))
            for i in range(0, len(features) - step, step)
        ]

    if not pairs:
        return CameraPath(
            centers=np.zeros((0, 3)), rotations=np.zeros((0, 3, 3)),
            reliable=False, note="no usable frame pairs",
        )

    centers = [np.zeros(3, dtype=np.float64)]
    rotations = [np.eye(3, dtype=np.float64)]
    used = 0
    for _, _, pair in pairs:
        r_world = rotations[-1]
        c_world = centers[-1]
        if pair.R is not None:
            r_rel = np.asarray(pair.R, dtype=np.float64)
            t_rel = (
                np.asarray(pair.t, dtype=np.float64).reshape(3)
                if pair.t is not None
                else np.zeros(3)
            )
            # Compose: world-from-camera rotation, and the new centre in world coords.
            r_new = r_world @ r_rel.T
            c_new = c_world - r_new @ t_rel
            used += 1
        else:
            # Pose estimation failed for this pair: stay put rather than invent motion.
            r_new, c_new = r_world, c_world
        rotations.append(r_new)
        centers.append(c_new)

    centers_arr = np.asarray(centers)
    rotations_arr = np.asarray(rotations)
    reliable = used >= max(2, int(0.5 * len(pairs)))
    note = "" if reliable else f"only {used} of {len(pairs)} pairs gave a pose"
    return CameraPath(
        centers=centers_arr, rotations=rotations_arr,
        reliable=reliable, note=note, n_steps_used=used,
    )


def _look_at_center(path: CameraPath) -> np.ndarray:
    """Where the cameras were pointing: the mean of their look-at points."""
    if path.n == 0:
        return np.zeros(3)
    distance = max(path.extent, 1e-6)
    forward = np.array([0.0, 0.0, 1.0])
    targets = np.array(
        [path.centers[i] + path.rotations[i] @ forward * distance for i in range(path.n)]
    )
    return targets.mean(axis=0)


def coverage_map(
    path: CameraPath,
    azimuth_bins: int | None = None,
    elevation_bins: int | None = None,
) -> Coverage:
    """Bin the camera positions by direction around the subject.

    Empty bins are counted only within the elevation rows the camera actually
    used: a capture filmed from one height is not punished for never flying, but
    it is punished for never getting to the back or the left.
    """
    cfg = get_config().labeling.coverage
    n_az = int(azimuth_bins or cfg.azimuth_bins)
    n_el = int(elevation_bins or cfg.elevation_bins)
    grid = np.zeros((n_az, n_el), dtype=np.int32)

    if path.n < 2 or path.extent <= 1e-6:
        return Coverage(
            grid=grid,
            empty_bins=n_az * n_el,
            empty_bins_all=n_az * n_el,
            empty_names=["every direction (no camera motion recovered)"],
            azimuth_span_deg=0.0,
            reliable=False,
        )

    center = _look_at_center(path)
    vectors = path.centers - center
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > 1e-9
    vectors, norms = vectors[keep], norms[keep]
    if vectors.shape[0] == 0:
        return Coverage(
            grid=grid, empty_bins=n_az * n_el, empty_bins_all=n_az * n_el,
            empty_names=["every direction"], azimuth_span_deg=0.0, reliable=False,
        )

    # Camera coordinates: +x right, +y down, +z forward, so "up" is -y.
    azimuth = np.degrees(np.arctan2(vectors[:, 0], vectors[:, 2])) % 360.0
    elevation = np.degrees(np.arcsin(np.clip(-vectors[:, 1] / norms, -1.0, 1.0)))

    az_idx = np.clip((azimuth / (360.0 / n_az)).astype(int), 0, n_az - 1)
    # Elevation bins spread over [-45, 45] degrees: beyond that is rare in practice.
    el_edges = np.linspace(-45.0, 45.0, n_el + 1)
    el_idx = np.clip(np.digitize(elevation, el_edges[1:-1]), 0, n_el - 1)
    for a, e in zip(az_idx, el_idx, strict=False):
        grid[a, e] += 1

    occupied_rows = [e for e in range(n_el) if grid[:, e].sum() > 0]
    empty_names: list[str] = []
    empty = 0
    for e in occupied_rows:
        for a in range(n_az):
            if grid[a, e] == 0:
                empty += 1
                name = AZIMUTH_NAMES[a] if n_az == len(AZIMUTH_NAMES) else f"az{a}"
                row = ELEVATION_NAMES[e] if n_el == len(ELEVATION_NAMES) else f"el{e}"
                empty_names.append(f"{name}, {row}")

    sorted_az = np.sort(azimuth)
    gaps = np.diff(np.concatenate([sorted_az, sorted_az[:1] + 360.0]))
    span = 360.0 - float(gaps.max()) if gaps.size else 0.0

    return Coverage(
        grid=grid,
        empty_bins=int(empty),
        empty_bins_all=int((grid == 0).sum()),
        empty_names=empty_names,
        azimuth_span_deg=span,
        reliable=path.reliable,
    )


def summarize_empty(coverage: Coverage, max_items: int = 3) -> str:
    """Short evidence string, e.g. ``"left, eye level; back, eye level"``."""
    if not coverage.empty_names:
        return ""
    shown = coverage.empty_names[:max_items]
    more = len(coverage.empty_names) - len(shown)
    text = "; ".join(shown)
    return f"{text} (+{more} more)" if more > 0 else text
