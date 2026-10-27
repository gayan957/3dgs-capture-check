"""Reproducible random seeds across random, numpy, cv2 and torch."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every random source we use."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # OpenCV's RNG (used by RANSAC)
        import cv2

        cv2.setRNGSeed(seed)
    except Exception:  # pragma: no cover - OpenCV always present in practice
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def rng(seed: int | None = None) -> np.random.Generator:
    """A numpy Generator, seeded when a seed is given."""
    return np.random.default_rng(seed)
