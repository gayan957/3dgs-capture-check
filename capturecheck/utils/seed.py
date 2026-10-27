"""Reproducible random seeds across random, numpy, cv2 and torch."""

from __future__ import annotations

import hashlib
import random

import numpy as np


def stable_seed(*parts: object, modulo: int = 2**31) -> int:
    """A seed derived from `parts` that is the same in every process.

    `hash()` of a string is salted per interpreter (PYTHONHASHSEED), so
    `abs(hash(name)) % n` picks a different number every run. Anything whose
    output has to be reproducible — a rendered demo scene, a broken copy — must
    derive its seed from a real digest instead.
    """
    digest = hashlib.blake2b(repr(parts).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def set_seed(seed: int, deterministic_torch: bool = True) -> None:
    """Seed every random source we use.

    This cannot make `hash()` of a string reproducible: PYTHONHASHSEED is read
    once at interpreter start-up, so setting it here would be too late to have
    any effect. Use `stable_seed` to derive a seed from text.
    """
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
