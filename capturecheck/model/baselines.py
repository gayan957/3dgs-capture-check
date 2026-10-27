"""Baselines the model has to beat, or be honestly compared against.

If a one-line heuristic ranks captures as well as a trained transformer, the
transformer is not earning its place, and the evaluation report should say so.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from capturecheck.model.data import GLOBAL_FEATURES, CaptureSample

#: Columns a LightGBM baseline is allowed to use: the same hand-made features the
#: transformer sees in its global vector, and nothing more.
TABULAR_FEATURES: tuple[str, ...] = GLOBAL_FEATURES


@dataclass
class BaselineResult:
    name: str
    description: str
    scores: np.ndarray
    note: str = ""


def _global_value(sample: CaptureSample, name: str) -> float:
    try:
        return float(sample.global_vec[GLOBAL_FEATURES.index(name)])
    except (ValueError, IndexError):
        return float("nan")


def frame_count_baseline(samples: list[CaptureSample]) -> BaselineResult:
    """More frames is better. The simplest thing anyone would try first."""
    return BaselineResult(
        name="frame_count",
        description="Number of frames or photos in the capture",
        scores=np.array([_global_value(s, "n_frames") for s in samples], dtype=float),
    )


def mean_blur_baseline(samples: list[CaptureSample]) -> BaselineResult:
    """Sharper frames are better: mean Laplacian variance over the capture."""
    return BaselineResult(
        name="mean_blur",
        description="Mean sharpness (variance of the Laplacian)",
        scores=np.array([_global_value(s, "sharpness_mean") for s in samples], dtype=float),
    )


def match_count_baseline(
    samples: list[CaptureSample], colmap_db: str | Path | None = None
) -> BaselineResult:
    """How well neighbouring frames match: the SfM-style baseline.

    With a COLMAP database the real two-view match counts are read from it. With
    no database the ORB overlap already measured stands in, which is weaker, and
    the report says so rather than passing it off as COLMAP.
    """
    if colmap_db is not None and Path(colmap_db).is_file():
        counts = _read_colmap_matches(Path(colmap_db), samples)
        if counts is not None:
            return BaselineResult(
                name="colmap_matches",
                description="Mean two-view match count from a COLMAP database",
                scores=counts,
            )

    overlap = np.array([_global_value(s, "overlap_mean") for s in samples], dtype=float)
    frames = np.array([_global_value(s, "n_frames") for s in samples], dtype=float)
    return BaselineResult(
        name="orb_matches",
        description="Mean neighbour overlap x frame count (ORB stand-in for COLMAP)",
        scores=overlap * frames,
        note=(
            "COLMAP was not available, so this is an ORB stand-in, not a COLMAP "
            "match count. Pass --colmap-db to use real COLMAP numbers."
        ),
    )


def _read_colmap_matches(db_path: Path, samples: list[CaptureSample]) -> np.ndarray | None:
    """Mean two-view match count per capture, read from a COLMAP database."""
    import sqlite3

    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        rows = connection.execute(
            "SELECT images.name, two_view_geometries.rows "
            "FROM two_view_geometries JOIN images "
            "ON images.image_id = two_view_geometries.pair_id / 2147483647"
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        connection.close()

    if not rows:
        return None
    per_capture: dict[str, list[float]] = {}
    for name, count in rows:
        capture_id = Path(str(name)).parent.name or Path(str(name)).stem
        per_capture.setdefault(capture_id, []).append(float(count))
    return np.array(
        [float(np.mean(per_capture.get(s.capture_id, [np.nan]))) for s in samples], dtype=float
    )


def lightgbm_baseline(
    train_samples: list[CaptureSample],
    test_samples: list[CaptureSample],
    seed: int = 0,
) -> BaselineResult:
    """LightGBM on the same hand-made features, trained to rank by damage/PSNR.

    This is the baseline that matters most: it uses exactly the capture-level
    numbers the transformer gets in its global vector, so any gap is down to the
    per-frame tokens and the pretext stage rather than to the features.
    """
    x_train = np.array([s.global_vec for s in train_samples], dtype=np.float64)
    x_test = np.array([s.global_vec for s in test_samples], dtype=np.float64)

    # Target: PSNR where known, otherwise "how undamaged" the capture is.
    target = []
    for sample in train_samples:
        if sample.has_psnr:
            target.append(sample.psnr)
        else:
            target.append(28.0 - 2.5 * sample.damage_rank)
    y_train = np.array(target, dtype=np.float64)

    note = ""
    try:
        import lightgbm as lgb

        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, subsample=0.9, colsample_bytree=0.9,
            random_state=seed, verbose=-1,
        )
        model.fit(np.nan_to_num(x_train), y_train)
        scores = model.predict(np.nan_to_num(x_test))
    except ImportError:  # pragma: no cover - lightgbm is in the dev extra
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(random_state=seed)
        model.fit(np.nan_to_num(x_train), y_train)
        scores = model.predict(np.nan_to_num(x_test))
        note = "LightGBM was not installed; scikit-learn gradient boosting was used instead."

    return BaselineResult(
        name="lightgbm_features",
        description="LightGBM on the capture-level features",
        scores=np.asarray(scores, dtype=float),
        note=note,
    )


def all_baselines(
    train_samples: list[CaptureSample],
    test_samples: list[CaptureSample],
    colmap_db: str | Path | None = None,
    seed: int = 0,
) -> list[BaselineResult]:
    """Every baseline, scored on the test samples."""
    builders: list[Callable[[], BaselineResult]] = [
        lambda: frame_count_baseline(test_samples),
        lambda: mean_blur_baseline(test_samples),
        lambda: match_count_baseline(test_samples, colmap_db),
        lambda: lightgbm_baseline(train_samples, test_samples, seed),
    ]
    return [build() for build in builders]


def as_frame(results: list[BaselineResult], samples: list[CaptureSample]) -> pd.DataFrame:
    data = {"capture_id": [s.capture_id for s in samples]}
    for result in results:
        data[result.name] = result.scores
    return pd.DataFrame(data)
