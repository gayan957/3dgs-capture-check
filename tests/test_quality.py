"""Frame quality and texture measurements."""

from __future__ import annotations

import time

import cv2
import numpy as np

from capturecheck.checks import quality, texture


def test_blur_lowers_sharpness(sharp_image):
    blurred = cv2.GaussianBlur(sharp_image, (0, 0), 5)
    assert quality.sharpness(blurred) < quality.sharpness(sharp_image)


def test_darkening_lowers_brightness(sharp_image):
    dark = (sharp_image * 0.3).astype(np.uint8)
    assert quality.brightness(dark) < quality.brightness(sharp_image)
    assert 0 <= quality.brightness(dark) <= 255


def test_clipping_detects_black_and_white_frames():
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    white = np.full((64, 64, 3), 255, dtype=np.uint8)
    assert quality.clipping(black) == (1.0, 0.0)
    assert quality.clipping(white) == (0.0, 1.0)


def test_noise_level_rises_with_added_noise(sharp_image):
    rng = np.random.default_rng(0)
    noisy = np.clip(
        sharp_image.astype(np.float64) + rng.normal(0, 20, sharp_image.shape), 0, 255
    ).astype(np.uint8)
    assert quality.noise_level(noisy) > quality.noise_level(sharp_image)


def test_flat_image_has_low_feature_count():
    flat = np.full((240, 320, 3), 128, dtype=np.uint8)
    assert texture.feature_count(flat) < 10


def test_textured_image_has_many_features(sharp_image):
    assert texture.feature_count(sharp_image) > 100


def test_edge_density_higher_for_texture(sharp_image):
    flat = np.full_like(sharp_image, 128)
    assert texture.edge_density(sharp_image) > texture.edge_density(flat)


def test_measure_returns_all_fields(sharp_image):
    out = quality.measure(sharp_image).as_dict()
    assert set(out) == {
        "sharpness", "brightness", "contrast", "clip_dark", "clip_bright", "noise",
    }
    assert all(np.isfinite(v) for v in out.values())


def test_quality_checks_are_fast():
    """A 960-px frame must stay well under 20 ms per measurement set."""
    frame = np.random.default_rng(1).integers(0, 255, (540, 960, 3), dtype=np.uint8)
    quality.measure(frame)  # warm up
    start = time.perf_counter()
    for _ in range(5):
        quality.measure(frame)
    per_frame = (time.perf_counter() - start) / 5
    assert per_frame < 0.05, f"{per_frame * 1000:.1f} ms per frame is too slow"
