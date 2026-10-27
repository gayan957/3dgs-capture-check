"""Builders shared between the model tests."""

from __future__ import annotations

import numpy as np

from capturecheck.labeling import schema
from capturecheck.model.data import FRAME_DIM, GLOBAL_DIM, PAIR_DIM, CaptureSample


def make_sample(
    capture_id: str = "cap",
    scene_id: str = "scene",
    n_frames: int = 12,
    n_pairs: int = 6,
    psnr: float = float("nan"),
    degrade_type: str = "none",
    degrade_strength: str = "",
    source: str = "real",
    input_type: str = "images",
    seed: int = 0,
) -> CaptureSample:
    """A random but well-formed :class:`CaptureSample`."""
    rng = np.random.default_rng(seed)
    return CaptureSample(
        capture_id=capture_id,
        scene_id=scene_id,
        frames=rng.normal(size=(n_frames, FRAME_DIM)).astype(np.float32),
        pairs=rng.normal(size=(n_pairs, PAIR_DIM)).astype(np.float32),
        global_vec=rng.normal(size=GLOBAL_DIM).astype(np.float32),
        problems=np.zeros(schema.N_PROBLEMS, dtype=np.float32),
        psnr=psnr,
        source=source,
        degrade_type=degrade_type,
        degrade_strength=degrade_strength,
        input_type=input_type,
    )
