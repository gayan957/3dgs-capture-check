"""How much texture a frame has: plain walls and skies cannot be reconstructed."""

from __future__ import annotations

import cv2
import numpy as np

from capturecheck.checks.quality import to_gray

#: Shared ORB detector. 2000 features is plenty at 960 px and keeps detection fast.
MAX_FEATURES = 2000
_ORB: cv2.ORB | None = None


def get_orb(max_features: int = MAX_FEATURES) -> cv2.ORB:
    """A reused ORB instance (creating one per frame is wasteful)."""
    global _ORB
    if _ORB is None or _ORB.getMaxFeatures() != max_features:
        _ORB = cv2.ORB_create(nfeatures=max_features, fastThreshold=12)
    return _ORB


def detect(img: np.ndarray, max_features: int = MAX_FEATURES):
    """ORB keypoints and descriptors for one frame."""
    gray = to_gray(img)
    orb = get_orb(max_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints or [], descriptors


def feature_count(img: np.ndarray, max_features: int = MAX_FEATURES) -> int:
    """Number of ORB keypoints: the cheapest proxy for "can COLMAP match this?"."""
    keypoints, _ = detect(img, max_features)
    return int(len(keypoints))


def feature_counts(frames: list[np.ndarray], max_features: int = MAX_FEATURES) -> list[int]:
    return [feature_count(f, max_features) for f in frames]


def edge_density(img: np.ndarray) -> float:
    """Share of pixels on a Canny edge: a second, detector-free texture view."""
    gray = to_gray(img)
    edges = cv2.Canny(gray, 60, 160)
    return float(np.count_nonzero(edges)) / float(edges.size)
