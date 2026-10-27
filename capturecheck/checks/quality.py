"""Per-frame image quality: sharpness, brightness, clipping and noise.

Pure functions on a BGR uint8 image. All of them must stay well under 20 ms for a
960-px frame on a laptop CPU, because they run on up to 600 frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

#: Pixel values at or below / above this count as clipped.
DARK_CUTOFF = 5
BRIGHT_CUTOFF = 250


def to_gray(img: np.ndarray) -> np.ndarray:
    """Greyscale view of a BGR (or already grey) image."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian: high for crisp edges, low for blur.

    Only comparable between frames measured at the same width, which is why the
    whole pipeline resizes to ``configs/labeling.yaml: resize_width`` first.
    """
    gray = to_gray(img)
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap.var())


def brightness(img: np.ndarray) -> float:
    """Mean grey level, 0 (black) to 255 (white)."""
    return float(to_gray(img).mean())


def contrast(img: np.ndarray) -> float:
    """Standard deviation of the grey level: low means flat, hazy or fogged."""
    return float(to_gray(img).std())


def clipping(img: np.ndarray) -> tuple[float, float]:
    """Share of pixels crushed to black and blown out to white, as ``(dark, bright)``."""
    gray = to_gray(img)
    total = float(gray.size)
    dark = float(np.count_nonzero(gray <= DARK_CUTOFF)) / total
    bright = float(np.count_nonzero(gray >= BRIGHT_CUTOFF)) / total
    return dark, bright


def noise_level(img: np.ndarray) -> float:
    """Noise estimate in grey levels.

    A 3x3 median filter removes the noise but keeps the structure, so what is left
    after subtracting it is mostly noise. The median absolute deviation of that
    residual is robust to the few strong edges that survive.
    """
    gray = to_gray(img)
    smooth = cv2.medianBlur(gray, 3)
    residual = gray.astype(np.int16) - smooth.astype(np.int16)
    mad = float(np.median(np.abs(residual)))
    return float(mad * 1.4826)  # scale a MAD to a Gaussian sigma


@dataclass
class FrameQuality:
    """Every quality measurement for one frame."""

    sharpness: float
    brightness: float
    contrast: float
    clip_dark: float
    clip_bright: float
    noise: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def measure(img: np.ndarray) -> FrameQuality:
    """All quality measurements for one frame, sharing a single greyscale conversion."""
    gray = to_gray(img)
    dark, bright = clipping(gray)
    return FrameQuality(
        sharpness=sharpness(gray),
        brightness=brightness(gray),
        contrast=contrast(gray),
        clip_dark=dark,
        clip_bright=bright,
        noise=noise_level(gray),
    )


def measure_many(frames: list[np.ndarray]) -> list[FrameQuality]:
    """Quality measurements for a whole capture."""
    return [measure(f) for f in frames]
