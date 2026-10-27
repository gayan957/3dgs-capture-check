"""Thresholds -> frame labels.

One row per frame in, the same rows with ``label_*`` columns out. Every threshold
comes from ``configs/labeling.yaml`` so the rules can be recalibrated for a new
camera without touching this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from capturecheck.config import LabelingConfig, labeling_config
from capturecheck.labeling import schema


def label_frames(measurements: pd.DataFrame, cfg: LabelingConfig | None = None) -> pd.DataFrame:
    """Add ``label_blur``, ``label_light``, ``label_view``, ``label_shiny``,
    ``label_texture`` and ``label_noise`` to a table of frame measurements."""
    cfg = cfg or labeling_config()
    df = measurements.copy()

    def column(name: str, default: float = np.nan) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
        return pd.Series(default, index=df.index, dtype=float)

    sharpness = column("sharpness")
    brightness = column("brightness")
    clip_dark = column("clip_dark", 0.0)
    clip_bright = column("clip_bright", 0.0)
    noise = column("noise")
    features = column("features")
    dup = column("dup_of_prev", 0.0)
    shiny = column("shiny_prob")

    # --- blur: a low Laplacian variance at a fixed width means a soft frame.
    df["label_blur"] = np.where(
        sharpness < cfg.blur.sharp_min_laplacian_var, "BLURRY", "SHARP"
    )
    df.loc[sharpness.isna(), "label_blur"] = "SHARP"

    # --- light: too dark, blown out, or crushed/clipped in either direction.
    # Clipping alone is not enough: a well-exposed, high-contrast frame can have a
    # black corner and a specular highlight without being badly lit. The overall
    # level has to lean the same way as the clipping before the frame is flagged.
    midpoint = (cfg.light.dark_mean + cfg.light.bright_mean) / 2.0
    too_dark = (brightness < cfg.light.dark_mean) | (
        (clip_dark > cfg.light.clip_dark_max) & (brightness < midpoint)
    )
    too_bright = (brightness > cfg.light.bright_mean) | (
        (clip_bright > cfg.light.clip_bright_max) & (brightness > midpoint)
    )
    light = np.full(len(df), "OK_LIGHT", dtype=object)
    light[too_bright.fillna(False).to_numpy()] = "BRIGHT"
    # Darkness wins when a frame is both crushed and clipped: it hurts more.
    light[too_dark.fillna(False).to_numpy()] = "DARK"
    df["label_light"] = light

    # --- view: near-identical to the last kept frame.
    df["label_view"] = np.where(
        dup > cfg.views.duplicate_cosine, "DUPLICATE", "UNIQUE_VIEW"
    )

    # --- shiny: only when a trained classifier gave us a probability.
    df["label_shiny"] = np.where(shiny >= cfg.shiny.prob, 1, 0).astype(int)
    df.loc[shiny.isna(), "label_shiny"] = 0

    # --- texture: too few ORB features to match anything.
    df["label_texture"] = np.where(
        features < cfg.texture.min_features, "LOW_TEXTURE", "OK_TEXTURE"
    )
    df.loc[features.isna(), "label_texture"] = "OK_TEXTURE"

    # --- noise: a high-ISO or very dark frame.
    df["label_noise"] = np.where(noise > cfg.noise.frame_max, 1, 0).astype(int)
    df.loc[noise.isna(), "label_noise"] = 0

    for name in schema.FRAME_LABEL_NAMES:
        if name not in df.columns:  # defensive: keep the schema complete
            df[name] = np.nan
    return df


def frame_label_counts(frame_df: pd.DataFrame) -> dict[str, int]:
    """How many frames carry each label value, for the CLI summary table."""
    counts: dict[str, int] = {}
    for definition in schema.FRAME_LABELS:
        if definition.name not in frame_df.columns:
            continue
        for value, count in frame_df[definition.name].value_counts().items():
            counts[f"{definition.name}={value}"] = int(count)
    return counts
