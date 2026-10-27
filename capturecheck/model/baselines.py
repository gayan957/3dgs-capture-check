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
from capturecheck.utils.logging import get_logger

log = get_logger(__name__)

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
    scores = _fit_lightgbm(np.nan_to_num(x_train), y_train, np.nan_to_num(x_test), seed)
    if scores is None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=5, random_state=seed,
        )
        model.fit(np.nan_to_num(x_train), y_train)
        scores = model.predict(np.nan_to_num(x_test))
        note = (
            "LightGBM could not be run here, so scikit-learn's histogram gradient "
            "boosting (the same family of model) was used instead."
        )

    return BaselineResult(
        name="lightgbm_features",
        description="Gradient-boosted trees on the capture-level features",
        scores=np.asarray(scores, dtype=float),
        note=note,
    )


def _fit_lightgbm(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int
) -> np.ndarray | None:
    """Fit LightGBM in a separate interpreter, or return ``None`` if that fails.

    LightGBM and PyTorch each bundle their own ``libomp`` on macOS, and a process
    that loads both segfaults mid-fit. CaptureCheck always imports torch, so
    LightGBM gets an interpreter of its own, which never does.
    """
    import subprocess
    import sys
    import tempfile

    with tempfile.TemporaryDirectory(prefix="capturecheck_lgbm_") as tmp:
        folder = Path(tmp)
        input_path = folder / "input.npz"
        output_path = folder / "scores.npy"
        np.savez(
            input_path,
            x_train=x_train.astype(np.float64),
            y_train=y_train.astype(np.float64),
            x_test=x_test.astype(np.float64),
            seed=np.array(seed),
        )
        command = [
            sys.executable, "-I", "-m", "capturecheck.model._lightgbm_worker",
            str(input_path), str(output_path),
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("could not start the LightGBM worker (%s)", exc)
            return None

        if completed.returncode != 0 or not output_path.is_file():
            log.warning(
                "the LightGBM worker failed (exit %s): %s",
                completed.returncode, (completed.stderr or "").strip()[-300:],
            )
            return None
        return np.load(output_path)


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
