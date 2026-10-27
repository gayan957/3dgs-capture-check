"""Turn measured captures into tensors, and generate feature-level damage.

A capture becomes three things:

* **frame tokens** -- one per embedded frame: the DINOv2 vector plus that frame's
  OpenCV measurements;
* **pair tokens** -- one per sampled frame pair: flow speed, walked confidence,
  brightness change;
* a **global vector** -- statistics over *all* frames (not just the embedded
  subset), the coverage grid and the file information.

Stage A needs damaged versions of these by the thousand. Rendering them as images
would cost hours, so the damage is applied to the *features* instead: lower the
sharpness numbers, drop a coverage sector, remove tokens. The model only ever
sees features, so feature-level damage is the same problem to learn from.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from capturecheck.config import Config, get_config
from capturecheck.features.dino import EMBED_DIM
from capturecheck.labeling import schema

#: Per-frame measurements that become part of a frame token.
FRAME_FEATURES: tuple[str, ...] = (
    "sharpness",
    "brightness",
    "contrast",
    "clip_dark",
    "clip_bright",
    "noise",
    "features",
    "overlap_next",
    "dup_of_prev",
    "shiny_prob",
)

#: Per-pair measurements that become a pair token.
PAIR_FEATURES: tuple[str, ...] = (
    "confidence",
    "walked",
    "n_matches",
    "inliers_h",
    "inliers_e",
    "homography_share",
    "brightness_delta",
    "flow_speed",
)

#: Capture-level statistics in the global vector, before the coverage grid.
GLOBAL_FEATURES: tuple[str, ...] = (
    *schema.VIDEO_STAT_COLUMNS,
    "n_frames",
    "n_cameras",
    "n_sizes",
    "is_video",
    "duration_s",
    "fps",
    "width",
    "height",
)

#: The damage types stage A learns to recognise (index 0 means "nothing wrong").
PRETEXT_TYPES: tuple[str, ...] = (
    "none",
    "blur",
    "drop_sector",
    "rotation_only",
    "subsample",
    "exposure_jitter",
    "noise",
    "drop_gap",
)

#: Which capture label each damage type implies.
PRETEXT_TO_LABEL: dict[str, str | None] = {
    "none": None,
    "blur": "motion_blur",
    "drop_sector": "poor_coverage",
    "rotation_only": "no_walking",
    "subsample": "too_few_views",
    "exposure_jitter": "lighting",
    "noise": "noisy",
    "drop_gap": "low_overlap",
}

FRAME_DIM = EMBED_DIM + len(FRAME_FEATURES) + 1  # +1 for the relative time
PAIR_DIM = len(PAIR_FEATURES) + 1  # +1 for the relative position
N_COVERAGE_BINS = 24
GLOBAL_DIM = len(GLOBAL_FEATURES) + N_COVERAGE_BINS


@dataclass
class CaptureSample:
    """One capture as plain numpy arrays, ready to be batched."""

    capture_id: str
    scene_id: str
    frames: np.ndarray  # (T, FRAME_DIM)
    pairs: np.ndarray  # (P, PAIR_DIM)
    global_vec: np.ndarray  # (GLOBAL_DIM,)
    problems: np.ndarray  # (K,) 0/1
    psnr: float = float("nan")
    source: str = "real"
    degrade_type: str = "none"
    degrade_strength: str = ""
    input_type: str = "video"

    @property
    def has_psnr(self) -> bool:
        return bool(np.isfinite(self.psnr))

    @property
    def damage_rank(self) -> float:
        """How broken this capture is, for the ranking loss: 0 = clean copy."""
        order = {"": 0.0, "light": 1.0, "medium": 2.0, "heavy": 3.0}
        if self.degrade_type in ("none", "") or self.source != "degraded":
            return 0.0
        return order.get(self.degrade_strength, 2.0)


@dataclass
class FeatureStats:
    """Mean and standard deviation used to normalise every input."""

    frame_mean: np.ndarray
    frame_std: np.ndarray
    pair_mean: np.ndarray
    pair_std: np.ndarray
    global_mean: np.ndarray
    global_std: np.ndarray

    @classmethod
    def fit(cls, samples: list[CaptureSample]) -> FeatureStats:
        def stack(arrays: list[np.ndarray], dim: int) -> np.ndarray:
            usable = [a for a in arrays if a.size]
            return np.vstack(usable) if usable else np.zeros((1, dim), dtype=np.float32)

        frames = stack([s.frames for s in samples], FRAME_DIM)
        pairs = stack([s.pairs for s in samples], PAIR_DIM)
        globals_ = stack([s.global_vec.reshape(1, -1) for s in samples], GLOBAL_DIM)

        def moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mean = np.nanmean(x, axis=0)
            std = np.nanstd(x, axis=0)
            mean = np.nan_to_num(mean, nan=0.0)
            std = np.nan_to_num(std, nan=1.0)
            std[std < 1e-6] = 1.0
            return mean.astype(np.float32), std.astype(np.float32)

        frame_mean, frame_std = moments(frames)
        pair_mean, pair_std = moments(pairs)
        global_mean, global_std = moments(globals_)
        return cls(frame_mean, frame_std, pair_mean, pair_std, global_mean, global_std)

    def normalise(self, sample: CaptureSample) -> CaptureSample:
        frames = (
            (sample.frames - self.frame_mean) / self.frame_std
            if sample.frames.size
            else sample.frames
        )
        pairs = (
            (sample.pairs - self.pair_mean) / self.pair_std if sample.pairs.size else sample.pairs
        )
        global_vec = (sample.global_vec - self.global_mean) / self.global_std
        return CaptureSample(
            capture_id=sample.capture_id,
            scene_id=sample.scene_id,
            frames=np.nan_to_num(frames, nan=0.0).astype(np.float32),
            pairs=np.nan_to_num(pairs, nan=0.0).astype(np.float32),
            global_vec=np.nan_to_num(global_vec, nan=0.0).astype(np.float32),
            problems=sample.problems,
            psnr=sample.psnr,
            source=sample.source,
            degrade_type=sample.degrade_type,
            degrade_strength=sample.degrade_strength,
            input_type=sample.input_type,
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        payload = {
            "frame_mean": self.frame_mean.tolist(),
            "frame_std": self.frame_std.tolist(),
            "pair_mean": self.pair_mean.tolist(),
            "pair_std": self.pair_std.tolist(),
            "global_mean": self.global_mean.tolist(),
            "global_std": self.global_std.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FeatureStats:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            **{k: np.asarray(v, dtype=np.float32) for k, v in payload.items()}
        )


def _column(df: pd.DataFrame, name: str, n: int) -> np.ndarray:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=np.float64)
    return np.full(n, np.nan)


def build_sample(
    capture_id: str,
    scene_id: str,
    frame_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    video_row: dict,
    emb: np.ndarray,
    emb_idx: np.ndarray,
    coverage_grid: np.ndarray | None = None,
    cfg: Config | None = None,
) -> CaptureSample:
    """Assemble one :class:`CaptureSample` from measured data."""
    cfg = cfg or get_config()
    max_tokens = cfg.train.model.max_tokens

    n_frames = len(frame_df)
    frame_df = frame_df.reset_index(drop=True)

    # --- frame tokens: the embedded subset, with that frame's measurements --- #
    if emb.size and emb_idx.size:
        idx = np.asarray(emb_idx, dtype=int)[: emb.shape[0]]
        vectors = emb[: idx.size]
    else:
        from capturecheck.features.dino import evenly_spaced

        idx = evenly_spaced(n_frames, max_tokens)
        vectors = np.zeros((idx.size, EMBED_DIM), dtype=np.float32)

    idx = idx[idx < n_frames]
    vectors = vectors[: idx.size]

    measurements = np.stack(
        [_column(frame_df, name, n_frames) for name in FRAME_FEATURES], axis=1
    )
    picked = measurements[idx] if idx.size else np.zeros((0, len(FRAME_FEATURES)))
    relative_time = (
        (idx / max(1, n_frames - 1)).reshape(-1, 1) if idx.size else np.zeros((0, 1))
    )
    frames = np.hstack([vectors, picked, relative_time]).astype(np.float32)
    if frames.shape[0] > max_tokens:
        frames = frames[:max_tokens]

    # --- pair tokens -------------------------------------------------------- #
    n_pairs = len(pair_df)
    if n_pairs:
        pair_df = pair_df.reset_index(drop=True)
        walked = (
            (pair_df["kind"] == "walked").to_numpy(dtype=float)
            if "kind" in pair_df.columns
            else np.full(n_pairs, np.nan)
        )
        columns = []
        for name in PAIR_FEATURES:
            columns.append(walked if name == "walked" else _column(pair_df, name, n_pairs))
        position = (np.arange(n_pairs) / max(1, n_pairs - 1)).reshape(-1, 1)
        pairs = np.hstack([np.stack(columns, axis=1), position]).astype(np.float32)
        if pairs.shape[0] > max_tokens:
            keep = np.linspace(0, pairs.shape[0] - 1, num=max_tokens).round().astype(int)
            pairs = pairs[keep]
    else:
        pairs = np.zeros((0, PAIR_DIM), dtype=np.float32)

    # --- global vector ------------------------------------------------------ #
    values = []
    for name in GLOBAL_FEATURES:
        if name == "is_video":
            values.append(1.0 if video_row.get("input_type") == "video" else 0.0)
            continue
        raw = video_row.get(name)
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            values.append(np.nan)

    grid = (
        np.asarray(coverage_grid, dtype=np.float64).reshape(-1)
        if coverage_grid is not None
        else np.full(N_COVERAGE_BINS, np.nan)
    )
    if grid.size < N_COVERAGE_BINS:
        grid = np.pad(grid, (0, N_COVERAGE_BINS - grid.size), constant_values=np.nan)
    grid = grid[:N_COVERAGE_BINS]
    # Occupied or not matters more than how many frames landed in a bin.
    grid = np.where(np.isfinite(grid), (grid > 0).astype(float), np.nan)
    global_vec = np.concatenate([np.asarray(values, dtype=np.float64), grid]).astype(np.float32)

    problems = np.array(
        [float(video_row.get(name, 0) or 0) for name in schema.PROBLEM_LABELS], dtype=np.float32
    )
    psnr = video_row.get("psnr")
    try:
        psnr = float(psnr)
    except (TypeError, ValueError):
        psnr = float("nan")

    return CaptureSample(
        capture_id=capture_id,
        scene_id=scene_id,
        frames=frames,
        pairs=pairs,
        global_vec=global_vec,
        problems=problems,
        psnr=psnr,
        source=str(video_row.get("source") or "real"),
        degrade_type=str(video_row.get("degrade_type") or "none"),
        degrade_strength=str(video_row.get("degrade_strength") or ""),
        input_type=str(video_row.get("input_type") or "video"),
    )


def samples_from_labels(
    labels_dir: str | Path | None = None,
    features_dir: str | Path | None = None,
    cfg: Config | None = None,
    scenes: list[str] | None = None,
) -> list[CaptureSample]:
    """Build samples from the label CSVs plus the cached DINOv2 features.

    ``capturecheck label`` must have run first. Captures with no cached
    embedding still work: their DINOv2 part is zeroed and only the OpenCV
    measurements carry information.
    """
    from capturecheck.io import dataset

    cfg = cfg or get_config()
    labels_dir = Path(labels_dir) if labels_dir else cfg.paths.resolve("labels")
    features_dir = Path(features_dir) if features_dir else cfg.paths.resolve("features")

    video_df = dataset.read_video_labels(out_dir=labels_dir)
    frame_df = dataset.read_frame_labels(out_dir=labels_dir)
    if video_df.empty:
        raise RuntimeError(f"no video_labels.csv in {labels_dir}; run `capturecheck label` first")
    video_df = dataset.merge_quality_targets(video_df, labels_dir)

    wanted = set(scenes) if scenes else None
    frames_by_capture = dict(tuple(frame_df.groupby("capture_id"))) if not frame_df.empty else {}

    out: list[CaptureSample] = []
    for _, row in video_df.iterrows():
        capture_id = str(row["capture_id"])
        scene_id = str(row.get("scene_id") or capture_id)
        if wanted is not None and scene_id not in wanted:
            continue

        rows = frames_by_capture.get(capture_id)
        if rows is None or rows.empty:
            continue

        emb = np.zeros((0, EMBED_DIM), dtype=np.float32)
        emb_idx = np.zeros(0, dtype=int)
        grid = None
        cache_path = features_dir / f"{capture_id}.npz"
        if cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as data:
                    if "emb" in data:
                        emb = data["emb"]
                        emb_idx = data["frame_idx"] if "frame_idx" in data else np.arange(len(emb))
                    if "coverage_grid" in data:
                        grid = data["coverage_grid"]
            except (OSError, ValueError):
                pass

        out.append(
            build_sample(
                capture_id=capture_id,
                scene_id=scene_id,
                frame_df=rows.sort_values("frame_idx"),
                pair_df=pd.DataFrame(),  # pair measurements are not in the CSVs
                video_row=row.to_dict(),
                emb=emb,
                emb_idx=emb_idx,
                coverage_grid=grid,
                cfg=cfg,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Stage A: feature-level damage
# --------------------------------------------------------------------------- #
@dataclass
class PretextSample:
    """A damaged copy of a clean capture, with the damage as its label."""

    sample: CaptureSample
    pretext_type: int
    strength: float
    problems: np.ndarray = field(default_factory=lambda: np.zeros(schema.N_PROBLEMS, np.float32))


class FeatureDegrader:
    """Makes damaged copies of clean captures at the feature level.

    Each op mirrors one of the image-level ops in :mod:`capturecheck.degrade.ops`,
    so what stage A learns transfers to real broken captures.
    """

    def __init__(self, stats: FeatureStats | None = None, seed: int = 0) -> None:
        self.stats = stats
        self.rng = np.random.default_rng(seed)
        self._frame_index = {name: i for i, name in enumerate(FRAME_FEATURES)}
        self._global_index = {name: i for i, name in enumerate(GLOBAL_FEATURES)}

    # Column positions inside a raw (un-normalised) frame token.
    def _frame_col(self, name: str) -> int:
        return EMBED_DIM + self._frame_index[name]

    def _global_col(self, name: str) -> int:
        return self._global_index[name]

    def apply(self, sample: CaptureSample, kind: str | None = None) -> PretextSample:
        """Damage one clean sample; ``kind=None`` picks an op at random."""
        choices = [k for k in PRETEXT_TYPES if k != "none"]
        if sample.input_type == "video":
            choices = [k for k in choices if k != "drop_gap"]
        kind = kind or str(self.rng.choice(choices))
        strength = float(self.rng.uniform(0.3, 1.0))

        frames = sample.frames.copy()
        pairs = sample.pairs.copy()
        global_vec = sample.global_vec.copy()
        problems = sample.problems.copy()

        label = PRETEXT_TO_LABEL.get(kind)
        if label is not None:
            problems[schema.PROBLEM_LABELS.index(label)] = 1.0

        if kind == "blur":
            share = 0.25 + 0.6 * strength
            n = frames.shape[0]
            count = max(1, int(round(share * n)))
            start = int(self.rng.integers(0, max(1, n - count + 1)))
            block = slice(start, start + count)
            frames[block, self._frame_col("sharpness")] *= 1.0 - 0.9 * strength
            frames[block, self._frame_col("contrast")] *= 1.0 - 0.3 * strength
            frames[block, self._frame_col("features")] *= 1.0 - 0.6 * strength
            frames[block, self._frame_col("overlap_next")] *= 1.0 - 0.4 * strength
            self._set_global(global_vec, "blur_ratio", share)
            self._scale_global(global_vec, "sharpness_mean", 1.0 - 0.7 * strength)

        elif kind == "drop_sector":
            n_bins = max(1, int(round(2 + 5 * strength)))
            start = int(self.rng.integers(0, N_COVERAGE_BINS))
            grid_start = len(GLOBAL_FEATURES)
            for offset in range(n_bins):
                global_vec[grid_start + (start + offset) % N_COVERAGE_BINS] = 0.0
            empty = float(np.count_nonzero(global_vec[grid_start:] == 0.0))
            self._set_global(global_vec, "coverage_empty_bins", empty)
            keep = self._keep_block(frames.shape[0], 1.0 - 0.3 * strength)
            frames = frames[keep]

        elif kind == "rotation_only":
            self._set_global(global_vec, "walked_ratio", max(0.0, 0.45 - 0.45 * strength))
            if pairs.size:
                pairs[:, PAIR_FEATURES.index("walked")] = 0.0
                pairs[:, PAIR_FEATURES.index("homography_share")] = 0.5
            self._set_global(global_vec, "coverage_empty_bins", float(N_COVERAGE_BINS))
            global_vec[len(GLOBAL_FEATURES) :] = 0.0

        elif kind == "subsample":
            keep_ratio = max(0.08, 1.0 - 0.85 * strength)
            n = frames.shape[0]
            count = max(2, int(round(keep_ratio * n)))
            keep = np.unique(np.linspace(0, n - 1, num=count).round().astype(int))
            frames = frames[keep]
            self._scale_global(global_vec, "n_frames", keep_ratio)
            self._scale_global(global_vec, "unique_views", keep_ratio)
            self._scale_global(global_vec, "overlap_mean", 1.0 - 0.5 * strength)

        elif kind == "exposure_jitter":
            noise = self.rng.normal(0.0, 40.0 * strength, size=frames.shape[0])
            frames[:, self._frame_col("brightness")] = np.clip(
                frames[:, self._frame_col("brightness")] + noise, 0, 255
            )
            frames[:, self._frame_col("clip_bright")] = np.clip(
                frames[:, self._frame_col("clip_bright")] + 0.2 * strength, 0, 1
            )
            self._set_global(global_vec, "brightness_std", 10.0 + 30.0 * strength)

        elif kind == "noise":
            frames[:, self._frame_col("noise")] += 10.0 * strength
            frames[:, self._frame_col("sharpness")] *= 1.0 + 0.2 * strength  # noise fakes detail
            self._set_global(global_vec, "noise_ratio", min(1.0, 0.3 + 0.7 * strength))
            self._scale_global(global_vec, "noise_mean", 1.0 + 2.0 * strength)

        elif kind == "drop_gap":
            n = frames.shape[0]
            count = max(1, int(round(0.1 * strength * n)))
            start = int(self.rng.integers(1, max(2, n - count)))
            frames = np.delete(frames, slice(start, start + count), axis=0)
            self._set_global(global_vec, "overlap_low_ratio", min(1.0, 0.2 + 0.5 * strength))
            self._scale_global(global_vec, "overlap_mean", 1.0 - 0.5 * strength)

        damaged = CaptureSample(
            capture_id=f"{sample.capture_id}__{kind}",
            scene_id=sample.scene_id,
            frames=frames.astype(np.float32),
            pairs=pairs.astype(np.float32),
            global_vec=global_vec.astype(np.float32),
            problems=problems,
            psnr=float("nan"),
            source="degraded",
            degrade_type=kind,
            degrade_strength=(
                "light" if strength < 0.5 else ("medium" if strength < 0.8 else "heavy")
            ),
            input_type=sample.input_type,
        )
        return PretextSample(
            sample=damaged,
            pretext_type=PRETEXT_TYPES.index(kind),
            strength=strength,
            problems=problems,
        )

    def _keep_block(self, n: int, keep_ratio: float) -> np.ndarray:
        count = max(2, int(round(np.clip(keep_ratio, 0.1, 1.0) * n)))
        start = int(self.rng.integers(0, max(1, n - count + 1)))
        return np.arange(start, min(n, start + count))

    def _set_global(self, vector: np.ndarray, name: str, value: float) -> None:
        if name in self._global_index:
            vector[self._global_col(name)] = value

    def _scale_global(self, vector: np.ndarray, name: str, factor: float) -> None:
        if name in self._global_index:
            vector[self._global_col(name)] *= factor


def clean_samples(samples: list[CaptureSample]) -> list[CaptureSample]:
    """The untouched captures: the only safe starting point for stage A."""
    return [s for s in samples if s.source == "real" and s.degrade_type in ("none", "")]
