"""Thresholds -> frame labels."""

from __future__ import annotations

import numpy as np
import pandas as pd

from capturecheck.config import get_config
from capturecheck.labeling.frame_rules import frame_label_counts, label_frames


def _measurements(**overrides) -> pd.DataFrame:
    """One well-behaved frame, with the named columns overridden."""
    base = {
        "sharpness": [500.0],
        "brightness": [120.0],
        "clip_dark": [0.0],
        "clip_bright": [0.0],
        "noise": [1.0],
        "features": [800],
        "dup_of_prev": [0.5],
        "shiny_prob": [0.1],
    }
    base.update({k: [v] for k, v in overrides.items()})
    return pd.DataFrame(base)


def test_good_frame_gets_every_ok_label():
    row = label_frames(_measurements()).iloc[0]
    assert row["label_blur"] == "SHARP"
    assert row["label_light"] == "OK_LIGHT"
    assert row["label_view"] == "UNIQUE_VIEW"
    assert row["label_shiny"] == 0
    assert row["label_texture"] == "OK_TEXTURE"
    assert row["label_noise"] == 0


def test_blur_threshold():
    cfg = get_config().labeling
    limit = cfg.blur.sharp_min_laplacian_var
    assert label_frames(_measurements(sharpness=limit - 1)).iloc[0]["label_blur"] == "BLURRY"
    assert label_frames(_measurements(sharpness=limit + 1)).iloc[0]["label_blur"] == "SHARP"


def test_light_labels():
    assert label_frames(_measurements(brightness=20)).iloc[0]["label_light"] == "DARK"
    assert label_frames(_measurements(brightness=230)).iloc[0]["label_light"] == "BRIGHT"
    # Clipping only counts when the overall level leans the same way.
    bright_with_dark_corner = _measurements(brightness=200, clip_dark=0.3)
    assert label_frames(bright_with_dark_corner).iloc[0]["label_light"] == "OK_LIGHT"
    dark_and_crushed = _measurements(brightness=60, clip_dark=0.3)
    assert label_frames(dark_and_crushed).iloc[0]["label_light"] == "DARK"


def test_duplicate_and_texture_and_noise():
    assert label_frames(_measurements(dup_of_prev=0.99)).iloc[0]["label_view"] == "DUPLICATE"
    assert label_frames(_measurements(features=10)).iloc[0]["label_texture"] == "LOW_TEXTURE"
    assert label_frames(_measurements(noise=20)).iloc[0]["label_noise"] == 1


def test_shiny_needs_a_probability():
    assert label_frames(_measurements(shiny_prob=0.9)).iloc[0]["label_shiny"] == 1
    # No classifier trained yet: NaN must not become a positive label.
    assert label_frames(_measurements(shiny_prob=np.nan)).iloc[0]["label_shiny"] == 0


def test_missing_columns_are_tolerated():
    df = label_frames(pd.DataFrame({"sharpness": [200.0]}))
    assert df.iloc[0]["label_blur"] == "SHARP"
    assert df.iloc[0]["label_light"] == "OK_LIGHT"


def test_frame_label_counts():
    df = label_frames(
        pd.DataFrame({"sharpness": [10.0, 500.0, 500.0], "brightness": [120.0] * 3})
    )
    counts = frame_label_counts(df)
    assert counts["label_blur=BLURRY"] == 1
    assert counts["label_blur=SHARP"] == 2
